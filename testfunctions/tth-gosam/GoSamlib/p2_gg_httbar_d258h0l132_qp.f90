module     p2_gg_httbar_d258h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d258h0l132_qp.f90
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
      use p2_gg_httbar_abbrevd258h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd258
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
      use p2_gg_httbar_abbrevd258h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(60) :: acd258
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd258(1)=dotproduct(k2,ninjaE3)
      acd258(2)=dotproduct(e1,ninjaE3)
      acd258(3)=dotproduct(ninjaE3,spvae2k2)
      acd258(4)=abb258(7)
      acd258(5)=dotproduct(e2,ninjaE3)
      acd258(6)=dotproduct(ninjaE3,spvae1k2)
      acd258(7)=abb258(57)
      acd258(8)=dotproduct(l4,ninjaE3)
      acd258(9)=abb258(122)
      acd258(10)=dotproduct(ninjaE3,spval4e2)
      acd258(11)=abb258(141)
      acd258(12)=dotproduct(ninjaE3,spval4e1)
      acd258(13)=abb258(17)
      acd258(14)=dotproduct(l5,ninjaE3)
      acd258(15)=dotproduct(ninjaE3,spval5e2)
      acd258(16)=abb258(112)
      acd258(17)=dotproduct(ninjaE3,spval5e1)
      acd258(18)=abb258(142)
      acd258(19)=dotproduct(ninjaA0,ninjaE3)
      acd258(20)=dotproduct(ninjaE3,spval4k2)
      acd258(21)=abb258(38)
      acd258(22)=dotproduct(ninjaE3,spval5k2)
      acd258(23)=abb258(54)
      acd258(24)=dotproduct(ninjaE3,spval4k1)
      acd258(25)=abb258(79)
      acd258(26)=dotproduct(ninjaE3,spval5k1)
      acd258(27)=abb258(84)
      acd258(28)=abb258(124)
      acd258(29)=abb258(9)
      acd258(30)=abb258(115)
      acd258(31)=dotproduct(ninjaE3,spvae2l4)
      acd258(32)=abb258(62)
      acd258(33)=dotproduct(ninjaE3,spvak2e2)
      acd258(34)=abb258(93)
      acd258(35)=abb258(125)
      acd258(36)=dotproduct(ninjaE3,spvae2l5)
      acd258(37)=dotproduct(ninjaE3,spvak1k2)
      acd258(38)=dotproduct(ninjaE3,spvae2k1)
      acd258(39)=dotproduct(ninjaE3,spvak1e2)
      acd258(40)=abb258(149)
      acd258(41)=abb258(152)
      acd258(42)=abb258(140)
      acd258(43)=dotproduct(ninjaE3,spvak2e1)
      acd258(44)=abb258(94)
      acd258(45)=dotproduct(ninjaE3,spvae1l4)
      acd258(46)=abb258(39)
      acd258(47)=dotproduct(ninjaE3,spvae1l5)
      acd258(48)=abb258(68)
      acd258(49)=dotproduct(ninjaE3,spvae1k1)
      acd258(50)=dotproduct(ninjaE3,spvak1e1)
      acd258(51)=2.0_ki*acd258(19)
      acd258(52)=acd258(51)-acd258(8)+acd258(14)
      acd258(52)=acd258(9)*acd258(52)
      acd258(53)=acd258(24)*acd258(25)
      acd258(54)=acd258(26)*acd258(27)
      acd258(55)=acd258(22)*acd258(23)
      acd258(56)=acd258(20)*acd258(21)
      acd258(52)=acd258(56)+acd258(55)+acd258(53)+acd258(54)+acd258(52)
      acd258(52)=acd258(5)*acd258(52)
      acd258(53)=acd258(15)*acd258(30)
      acd258(54)=acd258(10)*acd258(29)
      acd258(55)=acd258(28)*acd258(3)
      acd258(53)=acd258(55)+acd258(53)+acd258(54)
      acd258(53)=acd258(53)*acd258(51)
      acd258(54)=acd258(33)*acd258(35)
      acd258(55)=-acd258(28)*acd258(36)
      acd258(54)=acd258(54)+acd258(55)
      acd258(54)=acd258(22)*acd258(54)
      acd258(55)=-acd258(31)*acd258(32)
      acd258(56)=acd258(33)*acd258(34)
      acd258(55)=acd258(55)+acd258(56)
      acd258(55)=acd258(20)*acd258(55)
      acd258(56)=acd258(1)*acd258(3)*acd258(4)
      acd258(57)=acd258(28)*acd258(37)*acd258(38)
      acd258(58)=-acd258(26)*acd258(30)*acd258(39)
      acd258(59)=acd258(14)*acd258(15)*acd258(16)
      acd258(60)=acd258(8)*acd258(10)*acd258(11)
      acd258(52)=acd258(52)+acd258(53)+acd258(55)+acd258(54)+acd258(60)+acd258(&
      &59)+acd258(58)+acd258(56)+acd258(57)
      acd258(52)=acd258(2)*acd258(52)
      acd258(53)=acd258(17)*acd258(42)
      acd258(54)=acd258(12)*acd258(40)
      acd258(55)=-acd258(6)*acd258(41)
      acd258(53)=acd258(55)+acd258(53)+acd258(54)
      acd258(51)=acd258(53)*acd258(51)
      acd258(53)=-acd258(48)*acd258(47)
      acd258(54)=acd258(43)*acd258(46)
      acd258(53)=acd258(53)+acd258(54)
      acd258(53)=acd258(22)*acd258(53)
      acd258(54)=acd258(43)*acd258(44)
      acd258(55)=-acd258(41)*acd258(45)
      acd258(54)=acd258(54)+acd258(55)
      acd258(54)=acd258(20)*acd258(54)
      acd258(55)=acd258(37)*acd258(48)*acd258(49)
      acd258(56)=acd258(1)*acd258(6)*acd258(7)
      acd258(57)=-acd258(26)*acd258(42)*acd258(50)
      acd258(58)=acd258(14)*acd258(17)*acd258(18)
      acd258(59)=acd258(8)*acd258(12)*acd258(13)
      acd258(51)=acd258(51)+acd258(54)+acd258(53)+acd258(59)+acd258(58)+acd258(&
      &57)+acd258(55)+acd258(56)
      acd258(51)=acd258(5)*acd258(51)
      acd258(51)=acd258(51)+acd258(52)
      brack(ninjaidxt0x0mu0)=acd258(51)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d258h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd258h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
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
end module     p2_gg_httbar_d258h0l132_qp
