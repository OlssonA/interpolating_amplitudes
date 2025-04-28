module     p2_gg_httbar_d67h8l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d67h8l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd67h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd67
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd67h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(73) :: acd67
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd67(1)=dotproduct(k2,ninjaE3)
      acd67(2)=abb67(34)
      acd67(3)=dotproduct(ninjaE3,spvak1k2)
      acd67(4)=abb67(13)
      acd67(5)=dotproduct(ninjaE3,spvak1l5)
      acd67(6)=abb67(24)
      acd67(7)=dotproduct(ninjaE3,spvak2k1)
      acd67(8)=abb67(31)
      acd67(9)=dotproduct(ninjaE3,spval4k1)
      acd67(10)=abb67(30)
      acd67(11)=abb67(49)
      acd67(12)=dotproduct(ninjaA,ninjaE3)
      acd67(13)=abb67(48)
      acd67(14)=dotproduct(ninjaE3,spval4k2)
      acd67(15)=dotproduct(ninjaE3,spvak2l5)
      acd67(16)=abb67(35)
      acd67(17)=dotproduct(k2,ninjaA)
      acd67(18)=dotproduct(ninjaA,ninjaA)
      acd67(19)=abb67(12)
      acd67(20)=dotproduct(ninjaA,spvak1k2)
      acd67(21)=dotproduct(ninjaA,spvak1l5)
      acd67(22)=dotproduct(ninjaA,spvak2k1)
      acd67(23)=dotproduct(ninjaA,spval4k1)
      acd67(24)=abb67(10)
      acd67(25)=dotproduct(ninjaA,spval4k2)
      acd67(26)=dotproduct(ninjaA,spvak2l5)
      acd67(27)=dotproduct(ninjaE3,spval3k2)
      acd67(28)=abb67(11)
      acd67(29)=abb67(23)
      acd67(30)=dotproduct(ninjaE3,spval3l4)
      acd67(31)=abb67(14)
      acd67(32)=abb67(16)
      acd67(33)=dotproduct(ninjaE3,spval5k2)
      acd67(34)=abb67(17)
      acd67(35)=dotproduct(ninjaE3,spvak2l4)
      acd67(36)=abb67(18)
      acd67(37)=abb67(19)
      acd67(38)=dotproduct(ninjaE3,spval4l3)
      acd67(39)=abb67(20)
      acd67(40)=abb67(21)
      acd67(41)=dotproduct(ninjaE3,spvak2l3)
      acd67(42)=abb67(22)
      acd67(43)=dotproduct(ninjaE3,spval5l3)
      acd67(44)=abb67(26)
      acd67(45)=abb67(27)
      acd67(46)=abb67(28)
      acd67(47)=dotproduct(ninjaE3,spval3l5)
      acd67(48)=abb67(51)
      acd67(49)=acd67(2)*acd67(1)
      acd67(50)=acd67(4)*acd67(3)
      acd67(51)=acd67(6)*acd67(5)
      acd67(52)=acd67(8)*acd67(7)
      acd67(53)=acd67(10)*acd67(9)
      acd67(49)=acd67(53)+acd67(49)+acd67(50)+acd67(51)+acd67(52)
      acd67(50)=2.0_ki*acd67(12)
      acd67(51)=acd67(50)*acd67(49)
      acd67(52)=acd67(16)*acd67(9)
      acd67(53)=-acd67(5)*acd67(52)
      acd67(54)=acd67(13)*acd67(7)
      acd67(55)=acd67(3)*acd67(54)
      acd67(56)=acd67(15)*acd67(16)
      acd67(57)=acd67(14)*acd67(56)
      acd67(58)=acd67(11)*acd67(1)**2
      acd67(51)=acd67(58)+acd67(57)+acd67(55)+acd67(53)+acd67(51)
      acd67(53)=acd67(18)+ninjaP
      acd67(53)=acd67(49)*acd67(53)
      acd67(55)=acd67(2)*acd67(12)
      acd67(57)=acd67(11)*acd67(1)
      acd67(55)=acd67(55)+acd67(57)
      acd67(55)=acd67(17)*acd67(55)
      acd67(57)=acd67(4)*acd67(50)
      acd67(54)=acd67(57)+acd67(54)
      acd67(54)=acd67(20)*acd67(54)
      acd67(57)=acd67(6)*acd67(50)
      acd67(52)=-acd67(52)+acd67(57)
      acd67(52)=acd67(21)*acd67(52)
      acd67(57)=acd67(13)*acd67(3)
      acd67(58)=acd67(8)*acd67(50)
      acd67(57)=acd67(58)+acd67(57)
      acd67(57)=acd67(22)*acd67(57)
      acd67(58)=acd67(16)*acd67(5)
      acd67(59)=acd67(10)*acd67(50)
      acd67(58)=-acd67(58)+acd67(59)
      acd67(58)=acd67(23)*acd67(58)
      acd67(59)=acd67(26)*acd67(16)
      acd67(59)=acd67(32)+acd67(59)
      acd67(59)=acd67(14)*acd67(59)
      acd67(60)=acd67(19)*acd67(1)
      acd67(50)=acd67(24)*acd67(50)
      acd67(56)=acd67(25)*acd67(56)
      acd67(61)=acd67(28)*acd67(27)
      acd67(62)=acd67(29)*acd67(3)
      acd67(63)=acd67(31)*acd67(30)
      acd67(64)=acd67(34)*acd67(33)
      acd67(65)=acd67(36)*acd67(35)
      acd67(66)=acd67(37)*acd67(5)
      acd67(67)=acd67(39)*acd67(38)
      acd67(68)=acd67(40)*acd67(15)
      acd67(69)=acd67(42)*acd67(41)
      acd67(70)=acd67(44)*acd67(43)
      acd67(71)=acd67(45)*acd67(7)
      acd67(72)=acd67(46)*acd67(9)
      acd67(73)=acd67(48)*acd67(47)
      acd67(50)=acd67(73)+acd67(72)+acd67(71)+acd67(70)+acd67(69)+acd67(68)+acd&
      &67(67)+acd67(66)+acd67(65)+acd67(64)+acd67(63)+acd67(62)+acd67(61)+acd67&
      &(56)+acd67(50)+acd67(60)+acd67(58)+acd67(57)+acd67(52)+acd67(54)+2.0_ki*&
      &acd67(55)+acd67(53)+acd67(59)
      brack(ninjaidxt1mu0)=acd67(51)
      brack(ninjaidxt0mu0)=acd67(50)
      brack(ninjaidxt0mu2)=acd67(49)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d67h8_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd67h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k4
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d67h8l131_qp
