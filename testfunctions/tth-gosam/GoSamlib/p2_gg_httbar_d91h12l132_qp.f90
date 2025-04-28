module     p2_gg_httbar_d91h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d91h12l132_qp.f90
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
      use p2_gg_httbar_abbrevd91h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd91
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd91(1)=dotproduct(e2,ninjaE3)
      acd91(2)=dotproduct(ninjaE3,spvak2e1)
      acd91(3)=dotproduct(ninjaE3,spvae1l5)
      acd91(4)=abb91(14)
      acd91(5)=dotproduct(ninjaE3,spvae1l4)
      acd91(6)=abb91(22)
      acd91(7)=dotproduct(ninjaE3,spvae1l3)
      acd91(8)=abb91(26)
      acd91(9)=dotproduct(ninjaE3,spval3e1)
      acd91(10)=abb91(76)
      acd91(11)=acd91(4)*acd91(3)
      acd91(12)=acd91(6)*acd91(5)
      acd91(13)=acd91(8)*acd91(7)
      acd91(11)=acd91(13)+acd91(11)+acd91(12)
      acd91(11)=acd91(11)*acd91(2)
      acd91(12)=acd91(10)*acd91(9)*acd91(5)
      acd91(11)=acd91(12)+acd91(11)
      acd91(11)=acd91(1)*acd91(11)
      brack(ninjaidxt1x0mu0)=acd91(11)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd91h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(70) :: acd91
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd91(1)=dotproduct(e2,ninjaA1)
      acd91(2)=dotproduct(ninjaE3,spvae1l4)
      acd91(3)=dotproduct(ninjaE3,spvak2e1)
      acd91(4)=abb91(22)
      acd91(5)=dotproduct(ninjaE3,spval3e1)
      acd91(6)=abb91(76)
      acd91(7)=dotproduct(ninjaE3,spvae1l5)
      acd91(8)=abb91(14)
      acd91(9)=dotproduct(ninjaE3,spvae1l3)
      acd91(10)=abb91(26)
      acd91(11)=dotproduct(e2,ninjaE3)
      acd91(12)=dotproduct(ninjaA1,spvae1l4)
      acd91(13)=dotproduct(ninjaA1,spvak2e1)
      acd91(14)=dotproduct(ninjaA1,spvae1l5)
      acd91(15)=dotproduct(ninjaA1,spvae1l3)
      acd91(16)=dotproduct(ninjaA1,spval3e1)
      acd91(17)=dotproduct(e2,ninjaA0)
      acd91(18)=dotproduct(ninjaA0,spvae1l4)
      acd91(19)=dotproduct(ninjaA0,spvak2e1)
      acd91(20)=dotproduct(ninjaA0,spvae1l5)
      acd91(21)=dotproduct(ninjaA0,spvae1l3)
      acd91(22)=dotproduct(ninjaA0,spval3e1)
      acd91(23)=abb91(16)
      acd91(24)=abb91(18)
      acd91(25)=abb91(42)
      acd91(26)=abb91(47)
      acd91(27)=abb91(50)
      acd91(28)=dotproduct(ninjaA0,ninjaE3)
      acd91(29)=dotproduct(ninjaE3,spvae2e1)
      acd91(30)=abb91(49)
      acd91(31)=dotproduct(ninjaE3,spvae1e2)
      acd91(32)=abb91(57)
      acd91(33)=abb91(9)
      acd91(34)=abb91(15)
      acd91(35)=abb91(74)
      acd91(36)=abb91(23)
      acd91(37)=abb91(65)
      acd91(38)=dotproduct(ninjaE3,spval3k1)
      acd91(39)=abb91(29)
      acd91(40)=dotproduct(ninjaE3,spvak2k1)
      acd91(41)=abb91(31)
      acd91(42)=dotproduct(ninjaE3,spvae1k2)
      acd91(43)=abb91(44)
      acd91(44)=abb91(11)
      acd91(45)=abb91(12)
      acd91(46)=abb91(25)
      acd91(47)=dotproduct(ninjaE3,spvak1l5)
      acd91(48)=abb91(32)
      acd91(49)=dotproduct(ninjaE3,spvak1l3)
      acd91(50)=abb91(35)
      acd91(51)=abb91(37)
      acd91(52)=dotproduct(ninjaE3,spval4e1)
      acd91(53)=abb91(61)
      acd91(54)=acd91(8)*acd91(7)
      acd91(55)=acd91(10)*acd91(9)
      acd91(54)=acd91(54)+acd91(55)
      acd91(55)=acd91(3)*acd91(54)
      acd91(56)=acd91(2)*acd91(3)
      acd91(57)=acd91(56)*acd91(4)
      acd91(58)=acd91(6)*acd91(5)
      acd91(59)=acd91(58)*acd91(2)
      acd91(55)=acd91(59)+acd91(55)+acd91(57)
      acd91(57)=acd91(1)*acd91(55)
      acd91(54)=acd91(11)*acd91(54)
      acd91(59)=acd91(2)*acd91(11)
      acd91(60)=acd91(59)*acd91(4)
      acd91(54)=acd91(60)+acd91(54)
      acd91(60)=acd91(13)*acd91(54)
      acd91(61)=acd91(3)*acd91(11)
      acd91(62)=acd91(61)*acd91(4)
      acd91(58)=acd91(58)*acd91(11)
      acd91(58)=acd91(62)+acd91(58)
      acd91(62)=acd91(12)*acd91(58)
      acd91(63)=acd91(14)*acd91(8)*acd91(61)
      acd91(64)=acd91(61)*acd91(10)
      acd91(65)=acd91(15)*acd91(64)
      acd91(66)=acd91(59)*acd91(6)
      acd91(67)=acd91(16)*acd91(66)
      acd91(57)=acd91(67)+acd91(65)+acd91(63)+acd91(62)+acd91(57)+acd91(60)
      acd91(60)=2.0_ki*acd91(28)
      acd91(62)=acd91(30)*acd91(60)
      acd91(63)=acd91(33)*acd91(2)
      acd91(65)=acd91(36)*acd91(7)
      acd91(67)=acd91(37)*acd91(9)
      acd91(68)=acd91(39)*acd91(38)
      acd91(69)=acd91(41)*acd91(40)
      acd91(70)=acd91(43)*acd91(42)
      acd91(62)=acd91(70)+acd91(69)+acd91(68)+acd91(67)+acd91(65)+acd91(63)+acd&
      &91(62)
      acd91(62)=acd91(29)*acd91(62)
      acd91(60)=acd91(32)*acd91(60)
      acd91(63)=acd91(44)*acd91(3)
      acd91(65)=acd91(48)*acd91(47)
      acd91(67)=acd91(50)*acd91(49)
      acd91(68)=acd91(51)*acd91(5)
      acd91(69)=acd91(53)*acd91(52)
      acd91(60)=acd91(69)+acd91(68)+acd91(67)+acd91(65)+acd91(63)+acd91(60)
      acd91(60)=acd91(31)*acd91(60)
      acd91(63)=acd91(25)*acd91(7)
      acd91(65)=acd91(26)*acd91(9)
      acd91(67)=acd91(27)*acd91(5)
      acd91(63)=acd91(67)+acd91(65)+acd91(63)
      acd91(63)=acd91(11)*acd91(63)
      acd91(55)=acd91(17)*acd91(55)
      acd91(65)=acd91(45)*acd91(7)
      acd91(67)=acd91(46)*acd91(9)
      acd91(65)=acd91(67)+acd91(65)
      acd91(65)=acd91(3)*acd91(65)
      acd91(67)=acd91(20)*acd91(8)
      acd91(67)=acd91(24)+acd91(67)
      acd91(61)=acd91(61)*acd91(67)
      acd91(54)=acd91(19)*acd91(54)
      acd91(58)=acd91(18)*acd91(58)
      acd91(64)=acd91(21)*acd91(64)
      acd91(66)=acd91(22)*acd91(66)
      acd91(59)=acd91(23)*acd91(59)
      acd91(56)=acd91(34)*acd91(56)
      acd91(67)=acd91(35)*acd91(5)*acd91(2)
      acd91(54)=acd91(67)+acd91(56)+acd91(59)+acd91(66)+acd91(64)+acd91(58)+acd&
      &91(55)+acd91(54)+acd91(62)+acd91(60)+acd91(63)+acd91(61)+acd91(65)
      brack(ninjaidxt0x0mu0)=acd91(54)
      brack(ninjaidxt0x1mu0)=acd91(57)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d91h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd91h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k4
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d91h12l132_qp
