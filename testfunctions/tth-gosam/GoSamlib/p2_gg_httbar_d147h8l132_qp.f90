module     p2_gg_httbar_d147h8l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d147h8l132_qp.f90
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
      use p2_gg_httbar_abbrevd147h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd147
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd147(1)=dotproduct(ninjaE3,spvak2e2)
      acd147(2)=dotproduct(ninjaE3,spvae2k2)
      acd147(3)=abb147(12)
      acd147(4)=dotproduct(ninjaE3,spval4e2)
      acd147(5)=abb147(20)
      acd147(6)=dotproduct(ninjaE3,spvae1e2)
      acd147(7)=abb147(53)
      acd147(8)=dotproduct(ninjaE3,spvak1e2)
      acd147(9)=abb147(59)
      acd147(10)=dotproduct(ninjaE3,spval5e2)
      acd147(11)=abb147(61)
      acd147(12)=dotproduct(ninjaE3,spvae2k1)
      acd147(13)=abb147(13)
      acd147(14)=dotproduct(ninjaE3,spvae2e1)
      acd147(15)=abb147(14)
      acd147(16)=dotproduct(ninjaE3,spvae2l5)
      acd147(17)=abb147(22)
      acd147(18)=dotproduct(ninjaE3,spvae2l4)
      acd147(19)=abb147(28)
      acd147(20)=acd147(5)*acd147(2)
      acd147(21)=acd147(13)*acd147(12)
      acd147(22)=acd147(15)*acd147(14)
      acd147(23)=acd147(17)*acd147(16)
      acd147(24)=acd147(19)*acd147(18)
      acd147(20)=acd147(24)+acd147(23)+acd147(22)+acd147(21)+acd147(20)
      acd147(20)=acd147(4)*acd147(20)
      acd147(21)=acd147(3)*acd147(1)
      acd147(22)=acd147(7)*acd147(6)
      acd147(23)=-acd147(9)*acd147(8)
      acd147(24)=acd147(11)*acd147(10)
      acd147(21)=acd147(24)+acd147(23)+acd147(22)+acd147(21)
      acd147(21)=acd147(2)*acd147(21)
      acd147(20)=acd147(20)+acd147(21)
      brack(ninjaidxt1x0mu0)=acd147(20)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd147h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(62) :: acd147
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd147(1)=dotproduct(ninjaA1,spvak2e2)
      acd147(2)=dotproduct(ninjaE3,spvae2k2)
      acd147(3)=abb147(12)
      acd147(4)=dotproduct(ninjaA1,spvae2k2)
      acd147(5)=dotproduct(ninjaE3,spvak2e2)
      acd147(6)=dotproduct(ninjaE3,spval4e2)
      acd147(7)=abb147(20)
      acd147(8)=dotproduct(ninjaE3,spvak1e2)
      acd147(9)=abb147(59)
      acd147(10)=dotproduct(ninjaE3,spvae1e2)
      acd147(11)=abb147(53)
      acd147(12)=dotproduct(ninjaE3,spval5e2)
      acd147(13)=abb147(61)
      acd147(14)=dotproduct(ninjaA1,spvae2k1)
      acd147(15)=abb147(13)
      acd147(16)=dotproduct(ninjaA1,spval4e2)
      acd147(17)=dotproduct(ninjaE3,spvae2k1)
      acd147(18)=dotproduct(ninjaE3,spvae2e1)
      acd147(19)=abb147(14)
      acd147(20)=dotproduct(ninjaE3,spvae2l4)
      acd147(21)=abb147(28)
      acd147(22)=dotproduct(ninjaE3,spvae2l5)
      acd147(23)=abb147(22)
      acd147(24)=dotproduct(ninjaA1,spvae2e1)
      acd147(25)=dotproduct(ninjaA1,spvae2l4)
      acd147(26)=dotproduct(ninjaA1,spvae2l5)
      acd147(27)=dotproduct(ninjaA1,spvak1e2)
      acd147(28)=dotproduct(ninjaA1,spvae1e2)
      acd147(29)=dotproduct(ninjaA1,spval5e2)
      acd147(30)=dotproduct(ninjaA0,spvak2e2)
      acd147(31)=dotproduct(ninjaA0,spvae2k2)
      acd147(32)=dotproduct(ninjaA0,spvae2k1)
      acd147(33)=dotproduct(ninjaA0,spval4e2)
      acd147(34)=dotproduct(ninjaA0,spvae2e1)
      acd147(35)=dotproduct(ninjaA0,spvae2l4)
      acd147(36)=dotproduct(ninjaA0,spvae2l5)
      acd147(37)=dotproduct(ninjaA0,spvak1e2)
      acd147(38)=dotproduct(ninjaA0,spvae1e2)
      acd147(39)=dotproduct(ninjaA0,spval5e2)
      acd147(40)=abb147(21)
      acd147(41)=abb147(17)
      acd147(42)=abb147(19)
      acd147(43)=abb147(31)
      acd147(44)=abb147(18)
      acd147(45)=abb147(16)
      acd147(46)=abb147(29)
      acd147(47)=abb147(23)
      acd147(48)=abb147(25)
      acd147(49)=abb147(43)
      acd147(50)=acd147(23)*acd147(26)
      acd147(51)=acd147(21)*acd147(25)
      acd147(52)=acd147(19)*acd147(24)
      acd147(53)=acd147(15)*acd147(14)
      acd147(54)=acd147(4)*acd147(7)
      acd147(50)=acd147(54)+acd147(53)+acd147(52)+acd147(50)+acd147(51)
      acd147(50)=acd147(6)*acd147(50)
      acd147(51)=acd147(13)*acd147(29)
      acd147(52)=acd147(11)*acd147(28)
      acd147(53)=-acd147(9)*acd147(27)
      acd147(54)=acd147(3)*acd147(1)
      acd147(55)=acd147(16)*acd147(7)
      acd147(51)=acd147(55)+acd147(54)+acd147(53)+acd147(51)+acd147(52)
      acd147(51)=acd147(2)*acd147(51)
      acd147(52)=acd147(23)*acd147(22)
      acd147(53)=acd147(21)*acd147(20)
      acd147(54)=acd147(19)*acd147(18)
      acd147(55)=acd147(15)*acd147(17)
      acd147(52)=acd147(52)+acd147(53)+acd147(54)+acd147(55)
      acd147(53)=acd147(16)*acd147(52)
      acd147(54)=acd147(13)*acd147(12)
      acd147(55)=acd147(11)*acd147(10)
      acd147(56)=acd147(9)*acd147(8)
      acd147(57)=acd147(3)*acd147(5)
      acd147(54)=-acd147(54)-acd147(55)+acd147(56)-acd147(57)
      acd147(55)=-acd147(4)*acd147(54)
      acd147(50)=acd147(51)+acd147(50)+acd147(53)+acd147(55)
      acd147(51)=acd147(23)*acd147(36)
      acd147(53)=acd147(21)*acd147(35)
      acd147(55)=acd147(19)*acd147(34)
      acd147(56)=acd147(15)*acd147(32)
      acd147(57)=acd147(31)*acd147(7)
      acd147(51)=acd147(57)+acd147(56)+acd147(55)+acd147(53)+acd147(43)+acd147(&
      &51)
      acd147(51)=acd147(6)*acd147(51)
      acd147(53)=acd147(13)*acd147(39)
      acd147(55)=acd147(11)*acd147(38)
      acd147(56)=-acd147(9)*acd147(37)
      acd147(57)=acd147(3)*acd147(30)
      acd147(58)=acd147(33)*acd147(7)
      acd147(53)=acd147(58)+acd147(57)+acd147(56)+acd147(55)+acd147(41)+acd147(&
      &53)
      acd147(53)=acd147(2)*acd147(53)
      acd147(52)=acd147(33)*acd147(52)
      acd147(54)=-acd147(31)*acd147(54)
      acd147(55)=acd147(22)*acd147(46)
      acd147(56)=acd147(20)*acd147(45)
      acd147(57)=acd147(18)*acd147(44)
      acd147(58)=acd147(17)*acd147(42)
      acd147(59)=acd147(12)*acd147(49)
      acd147(60)=acd147(10)*acd147(48)
      acd147(61)=acd147(8)*acd147(47)
      acd147(62)=acd147(5)*acd147(40)
      acd147(51)=acd147(53)+acd147(51)+acd147(54)+acd147(52)+acd147(62)+acd147(&
      &61)+acd147(60)+acd147(59)+acd147(58)+acd147(57)+acd147(55)+acd147(56)
      brack(ninjaidxt0x0mu0)=acd147(51)
      brack(ninjaidxt0x1mu0)=acd147(50)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d147h8_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd147h8_qp
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
end module     p2_gg_httbar_d147h8l132_qp
