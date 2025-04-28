module     p2_gg_httbar_d130h8l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d130h8l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd130h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd130
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd130h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(63) :: acd130
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd130(1)=dotproduct(k2,ninjaE3)
      acd130(2)=abb130(21)
      acd130(3)=dotproduct(l3,ninjaE3)
      acd130(4)=abb130(86)
      acd130(5)=dotproduct(l4,ninjaE3)
      acd130(6)=abb130(39)
      acd130(7)=dotproduct(ninjaA0,ninjaE3)
      acd130(8)=abb130(26)
      acd130(9)=dotproduct(ninjaE3,spval4k2)
      acd130(10)=abb130(13)
      acd130(11)=dotproduct(ninjaE3,spvae1k2)
      acd130(12)=abb130(14)
      acd130(13)=dotproduct(ninjaE3,spvak2l3)
      acd130(14)=abb130(15)
      acd130(15)=dotproduct(ninjaE3,spval4l3)
      acd130(16)=abb130(16)
      acd130(17)=dotproduct(ninjaE3,spvak1k2)
      acd130(18)=abb130(17)
      acd130(19)=dotproduct(ninjaE3,spval4k1)
      acd130(20)=abb130(18)
      acd130(21)=dotproduct(ninjaE3,spval4e1)
      acd130(22)=abb130(20)
      acd130(23)=dotproduct(ninjaE3,spval3l4)
      acd130(24)=abb130(22)
      acd130(25)=dotproduct(ninjaE3,spval3k2)
      acd130(26)=abb130(23)
      acd130(27)=dotproduct(ninjaE3,spvae2k2)
      acd130(28)=abb130(27)
      acd130(29)=dotproduct(ninjaE3,spval4e2)
      acd130(30)=abb130(29)
      acd130(31)=dotproduct(ninjaE3,spvae2l3)
      acd130(32)=abb130(34)
      acd130(33)=dotproduct(ninjaE3,spvae1l3)
      acd130(34)=abb130(36)
      acd130(35)=dotproduct(ninjaE3,spval3e2)
      acd130(36)=abb130(40)
      acd130(37)=dotproduct(ninjaE3,spval3e1)
      acd130(38)=abb130(41)
      acd130(39)=dotproduct(ninjaE3,spvak1l3)
      acd130(40)=abb130(161)
      acd130(41)=dotproduct(ninjaE3,spval3k1)
      acd130(42)=abb130(179)
      acd130(43)=acd130(2)*acd130(1)
      acd130(44)=acd130(4)*acd130(3)
      acd130(45)=acd130(6)*acd130(5)
      acd130(46)=acd130(8)*acd130(7)
      acd130(47)=acd130(10)*acd130(9)
      acd130(48)=acd130(12)*acd130(11)
      acd130(49)=acd130(14)*acd130(13)
      acd130(50)=acd130(16)*acd130(15)
      acd130(51)=acd130(18)*acd130(17)
      acd130(52)=acd130(20)*acd130(19)
      acd130(53)=acd130(22)*acd130(21)
      acd130(54)=acd130(24)*acd130(23)
      acd130(55)=acd130(26)*acd130(25)
      acd130(56)=acd130(28)*acd130(27)
      acd130(57)=acd130(30)*acd130(29)
      acd130(58)=acd130(32)*acd130(31)
      acd130(59)=acd130(34)*acd130(33)
      acd130(60)=acd130(36)*acd130(35)
      acd130(61)=acd130(38)*acd130(37)
      acd130(62)=-acd130(40)*acd130(39)
      acd130(63)=-acd130(42)*acd130(41)
      acd130(43)=acd130(63)+acd130(62)+acd130(61)+acd130(60)+acd130(59)+acd130(&
      &58)+acd130(57)+acd130(56)+acd130(55)+acd130(54)+acd130(53)+acd130(52)+ac&
      &d130(51)+acd130(50)+acd130(49)+acd130(48)+acd130(47)+2.0_ki*acd130(46)+a&
      &cd130(45)+acd130(43)+acd130(44)
      brack(ninjaidxt0x0mu0)=acd130(43)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d130h8_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd130h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k4
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d130h8l132_qp
