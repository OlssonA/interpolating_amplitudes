module     p2_gg_httbar_d29h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d29h4l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd29h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd29
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd29h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(108) :: acd29
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd29(1)=dotproduct(k2,ninjaE3)
      acd29(2)=abb29(14)
      acd29(3)=dotproduct(ninjaA0,ninjaE3)
      acd29(4)=abb29(17)
      acd29(5)=dotproduct(ninjaE3,spvak2e2)
      acd29(6)=abb29(11)
      acd29(7)=dotproduct(ninjaE3,spvae1k2)
      acd29(8)=abb29(12)
      acd29(9)=dotproduct(ninjaE3,spval3k2)
      acd29(10)=abb29(15)
      acd29(11)=dotproduct(ninjaE3,spvak2l4)
      acd29(12)=abb29(16)
      acd29(13)=dotproduct(ninjaE3,spval3k1)
      acd29(14)=abb29(18)
      acd29(15)=dotproduct(ninjaE3,spvae2k1)
      acd29(16)=abb29(19)
      acd29(17)=dotproduct(ninjaE3,spval3l5)
      acd29(18)=abb29(20)
      acd29(19)=dotproduct(ninjaE3,spval3l4)
      acd29(20)=abb29(21)
      acd29(21)=dotproduct(ninjaE3,spvae1e2)
      acd29(22)=abb29(22)
      acd29(23)=dotproduct(ninjaE3,spvae2l5)
      acd29(24)=abb29(23)
      acd29(25)=dotproduct(ninjaE3,spvak2l3)
      acd29(26)=abb29(24)
      acd29(27)=dotproduct(ninjaE3,spvak2l5)
      acd29(28)=abb29(25)
      acd29(29)=dotproduct(ninjaE3,spvak1e2)
      acd29(30)=abb29(26)
      acd29(31)=dotproduct(ninjaE3,spvak2k1)
      acd29(32)=abb29(27)
      acd29(33)=dotproduct(ninjaE3,spvak1k2)
      acd29(34)=abb29(28)
      acd29(35)=dotproduct(ninjaE3,spvak1l4)
      acd29(36)=abb29(29)
      acd29(37)=dotproduct(ninjaE3,spval5e2)
      acd29(38)=abb29(30)
      acd29(39)=dotproduct(ninjaE3,spvae1l4)
      acd29(40)=abb29(31)
      acd29(41)=dotproduct(ninjaE3,spvak1l3)
      acd29(42)=abb29(32)
      acd29(43)=dotproduct(ninjaE3,spval5k2)
      acd29(44)=abb29(33)
      acd29(45)=dotproduct(ninjaE3,spval5l4)
      acd29(46)=abb29(34)
      acd29(47)=dotproduct(ninjaE3,spval4e1)
      acd29(48)=abb29(35)
      acd29(49)=dotproduct(ninjaE3,spval4k1)
      acd29(50)=abb29(37)
      acd29(51)=dotproduct(ninjaE3,spvae2l3)
      acd29(52)=abb29(39)
      acd29(53)=dotproduct(ninjaE3,spval5l3)
      acd29(54)=abb29(40)
      acd29(55)=dotproduct(ninjaE3,spval3e2)
      acd29(56)=abb29(42)
      acd29(57)=dotproduct(ninjaE3,spvae2k2)
      acd29(58)=abb29(43)
      acd29(59)=dotproduct(ninjaE3,spval4l5)
      acd29(60)=abb29(44)
      acd29(61)=dotproduct(ninjaE3,spval4l3)
      acd29(62)=abb29(45)
      acd29(63)=dotproduct(ninjaE3,spval4k2)
      acd29(64)=abb29(49)
      acd29(65)=dotproduct(ninjaE3,spvae2e1)
      acd29(66)=abb29(52)
      acd29(67)=dotproduct(ninjaE3,spval3e1)
      acd29(68)=abb29(57)
      acd29(69)=dotproduct(ninjaE3,spvak2e1)
      acd29(70)=abb29(72)
      acd29(71)=dotproduct(ninjaE3,spvae1l3)
      acd29(72)=abb29(73)
      acd29(73)=acd29(2)*acd29(1)
      acd29(74)=acd29(4)*acd29(3)
      acd29(75)=acd29(6)*acd29(5)
      acd29(76)=acd29(8)*acd29(7)
      acd29(77)=acd29(10)*acd29(9)
      acd29(78)=acd29(12)*acd29(11)
      acd29(79)=acd29(14)*acd29(13)
      acd29(80)=acd29(16)*acd29(15)
      acd29(81)=acd29(18)*acd29(17)
      acd29(82)=acd29(20)*acd29(19)
      acd29(83)=acd29(22)*acd29(21)
      acd29(84)=acd29(24)*acd29(23)
      acd29(85)=acd29(26)*acd29(25)
      acd29(86)=acd29(28)*acd29(27)
      acd29(87)=acd29(30)*acd29(29)
      acd29(88)=acd29(32)*acd29(31)
      acd29(89)=acd29(34)*acd29(33)
      acd29(90)=acd29(36)*acd29(35)
      acd29(91)=acd29(38)*acd29(37)
      acd29(92)=acd29(40)*acd29(39)
      acd29(93)=acd29(42)*acd29(41)
      acd29(94)=acd29(44)*acd29(43)
      acd29(95)=acd29(46)*acd29(45)
      acd29(96)=acd29(48)*acd29(47)
      acd29(97)=acd29(50)*acd29(49)
      acd29(98)=acd29(52)*acd29(51)
      acd29(99)=acd29(54)*acd29(53)
      acd29(100)=acd29(56)*acd29(55)
      acd29(101)=acd29(58)*acd29(57)
      acd29(102)=acd29(60)*acd29(59)
      acd29(103)=acd29(62)*acd29(61)
      acd29(104)=acd29(64)*acd29(63)
      acd29(105)=acd29(66)*acd29(65)
      acd29(106)=-acd29(68)*acd29(67)
      acd29(107)=-acd29(70)*acd29(69)
      acd29(108)=acd29(72)*acd29(71)
      acd29(73)=acd29(108)+acd29(107)+acd29(106)+acd29(105)+acd29(104)+acd29(10&
      &3)+acd29(102)+acd29(101)+acd29(100)+acd29(99)+acd29(98)+acd29(97)+acd29(&
      &96)+acd29(95)+acd29(94)+acd29(93)+acd29(92)+acd29(91)+acd29(90)+acd29(89&
      &)+acd29(88)+acd29(87)+acd29(86)+acd29(85)+acd29(84)+acd29(83)+acd29(82)+&
      &acd29(81)+acd29(80)+acd29(79)+acd29(78)+acd29(77)+acd29(76)+acd29(75)+ac&
      &d29(73)+2.0_ki*acd29(74)
      brack(ninjaidxt0x0mu0)=acd29(73)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d29h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd29h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4
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
end module     p2_gg_httbar_d29h4l132
