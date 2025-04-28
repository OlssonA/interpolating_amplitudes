module     p2_gg_httbar_d261h8l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d261h8l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd261h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd261
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd261h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(96) :: acd261
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd261(1)=dotproduct(ninjaE3,spval4e1)
      acd261(2)=dotproduct(ninjaE3,spvae2l5)
      acd261(3)=dotproduct(ninjaE3,spvae1e2)
      acd261(4)=abb261(8)
      acd261(5)=dotproduct(ninjaE3,spvae1k2)
      acd261(6)=dotproduct(ninjaE3,spvak2e2)
      acd261(7)=dotproduct(ninjaE3,spvae2e1)
      acd261(8)=abb261(17)
      acd261(9)=dotproduct(k2,ninjaE3)
      acd261(10)=dotproduct(ninjaA,ninjaE3)
      acd261(11)=abb261(44)
      acd261(12)=dotproduct(ninjaE3,spvae1l3)
      acd261(13)=abb261(68)
      acd261(14)=dotproduct(ninjaE3,spval3e1)
      acd261(15)=abb261(61)
      acd261(16)=abb261(25)
      acd261(17)=dotproduct(ninjaE3,spvae2k1)
      acd261(18)=abb261(43)
      acd261(19)=dotproduct(ninjaE3,spvak2e1)
      acd261(20)=abb261(53)
      acd261(21)=abb261(58)
      acd261(22)=abb261(51)
      acd261(23)=abb261(40)
      acd261(24)=dotproduct(ninjaE3,spvak2l5)
      acd261(25)=abb261(13)
      acd261(26)=abb261(47)
      acd261(27)=abb261(29)
      acd261(28)=dotproduct(ninjaE3,spvak2l3)
      acd261(29)=abb261(19)
      acd261(30)=dotproduct(ninjaE3,spval3k2)
      acd261(31)=abb261(24)
      acd261(32)=dotproduct(ninjaE3,spval4k2)
      acd261(33)=abb261(36)
      acd261(34)=abb261(87)
      acd261(35)=dotproduct(ninjaE3,spvak1e2)
      acd261(36)=abb261(42)
      acd261(37)=dotproduct(ninjaE3,spval4l3)
      acd261(38)=abb261(66)
      acd261(39)=dotproduct(ninjaE3,spval3l5)
      acd261(40)=abb261(73)
      acd261(41)=dotproduct(ninjaA,spval4e1)
      acd261(42)=dotproduct(ninjaA,spvae2l5)
      acd261(43)=dotproduct(ninjaA,spvae1e2)
      acd261(44)=dotproduct(ninjaA,spvae1k2)
      acd261(45)=dotproduct(ninjaA,spvak2e2)
      acd261(46)=dotproduct(ninjaA,spvae2e1)
      acd261(47)=abb261(7)
      acd261(48)=abb261(52)
      acd261(49)=abb261(35)
      acd261(50)=abb261(62)
      acd261(51)=abb261(70)
      acd261(52)=abb261(57)
      acd261(53)=abb261(76)
      acd261(54)=abb261(9)
      acd261(55)=abb261(59)
      acd261(56)=abb261(55)
      acd261(57)=abb261(56)
      acd261(58)=dotproduct(ninjaE3,spvae2k2)
      acd261(59)=abb261(37)
      acd261(60)=abb261(45)
      acd261(61)=abb261(31)
      acd261(62)=abb261(77)
      acd261(63)=abb261(48)
      acd261(64)=abb261(74)
      acd261(65)=abb261(27)
      acd261(66)=abb261(63)
      acd261(67)=abb261(28)
      acd261(68)=abb261(65)
      acd261(69)=abb261(46)
      acd261(70)=abb261(72)
      acd261(71)=abb261(32)
      acd261(72)=dotproduct(ninjaE3,spval4e2)
      acd261(73)=abb261(30)
      acd261(74)=abb261(38)
      acd261(75)=abb261(71)
      acd261(76)=abb261(69)
      acd261(77)=acd261(1)*acd261(2)*acd261(3)*acd261(4)
      acd261(78)=acd261(6)*acd261(7)
      acd261(79)=acd261(5)*acd261(8)*acd261(78)
      acd261(77)=acd261(77)+acd261(79)
      acd261(79)=acd261(32)*acd261(33)
      acd261(80)=acd261(24)*acd261(25)
      acd261(81)=acd261(39)*acd261(40)
      acd261(82)=acd261(37)*acd261(38)
      acd261(83)=acd261(35)*acd261(36)
      acd261(84)=acd261(30)*acd261(31)
      acd261(85)=acd261(28)*acd261(29)
      acd261(86)=acd261(17)*acd261(18)
      acd261(87)=acd261(9)*acd261(11)
      acd261(88)=acd261(19)*acd261(20)
      acd261(89)=acd261(6)*acd261(27)
      acd261(90)=acd261(14)*acd261(34)
      acd261(91)=acd261(12)*acd261(23)
      acd261(92)=acd261(2)*acd261(22)
      acd261(93)=acd261(1)*acd261(21)
      acd261(94)=acd261(5)*acd261(26)
      acd261(95)=2.0_ki*acd261(10)
      acd261(96)=acd261(16)*acd261(95)
      acd261(79)=acd261(96)+acd261(94)+acd261(93)+acd261(92)+acd261(91)+acd261(&
      &90)+acd261(89)+acd261(88)+acd261(87)+acd261(86)+acd261(85)+acd261(84)+ac&
      &d261(83)+acd261(82)+acd261(81)+acd261(79)+acd261(80)
      acd261(79)=acd261(79)*acd261(95)
      acd261(80)=acd261(39)*acd261(70)
      acd261(81)=acd261(35)*acd261(69)
      acd261(82)=acd261(30)*acd261(67)
      acd261(83)=acd261(8)*acd261(45)
      acd261(83)=acd261(66)+acd261(83)
      acd261(83)=acd261(7)*acd261(83)
      acd261(84)=acd261(8)*acd261(46)
      acd261(84)=acd261(65)+acd261(84)
      acd261(84)=acd261(6)*acd261(84)
      acd261(85)=acd261(14)*acd261(68)
      acd261(86)=acd261(1)*acd261(55)
      acd261(80)=acd261(86)+acd261(85)+acd261(84)+acd261(83)+acd261(82)+acd261(&
      &80)+acd261(81)
      acd261(80)=acd261(5)*acd261(80)
      acd261(81)=acd261(24)*acd261(60)
      acd261(82)=acd261(39)*acd261(64)
      acd261(83)=acd261(35)*acd261(63)
      acd261(84)=acd261(30)*acd261(62)
      acd261(85)=acd261(9)*acd261(13)
      acd261(86)=acd261(6)*acd261(61)
      acd261(81)=acd261(86)+acd261(85)+acd261(84)+acd261(83)+acd261(81)+acd261(&
      &82)
      acd261(81)=acd261(12)*acd261(81)
      acd261(82)=acd261(32)*acd261(75)
      acd261(83)=acd261(37)*acd261(76)
      acd261(84)=acd261(28)*acd261(74)
      acd261(85)=acd261(17)*acd261(48)
      acd261(86)=acd261(9)*acd261(15)
      acd261(82)=acd261(86)+acd261(85)+acd261(84)+acd261(82)+acd261(83)
      acd261(82)=acd261(14)*acd261(82)
      acd261(83)=acd261(37)*acd261(51)
      acd261(84)=acd261(28)*acd261(50)
      acd261(85)=acd261(17)*acd261(47)
      acd261(83)=acd261(85)+acd261(83)+acd261(84)
      acd261(83)=acd261(19)*acd261(83)
      acd261(84)=acd261(19)*acd261(49)
      acd261(85)=acd261(4)*acd261(41)
      acd261(85)=acd261(56)+acd261(85)
      acd261(85)=acd261(3)*acd261(85)
      acd261(86)=acd261(14)*acd261(57)
      acd261(84)=acd261(86)+acd261(84)+acd261(85)
      acd261(84)=acd261(2)*acd261(84)
      acd261(85)=acd261(4)*acd261(42)
      acd261(85)=acd261(53)+acd261(85)
      acd261(85)=acd261(3)*acd261(85)
      acd261(86)=acd261(12)*acd261(54)
      acd261(87)=acd261(4)*acd261(43)
      acd261(87)=acd261(52)+acd261(87)
      acd261(87)=acd261(2)*acd261(87)
      acd261(85)=acd261(87)+acd261(85)+acd261(86)
      acd261(85)=acd261(1)*acd261(85)
      acd261(86)=acd261(8)*acd261(44)
      acd261(86)=acd261(71)+acd261(86)
      acd261(78)=acd261(86)*acd261(78)
      acd261(86)=acd261(7)*acd261(72)*acd261(73)
      acd261(87)=acd261(3)*acd261(58)*acd261(59)
      acd261(78)=acd261(79)+acd261(80)+acd261(85)+acd261(84)+acd261(81)+acd261(&
      &82)+acd261(78)+acd261(87)+acd261(83)+acd261(86)
      brack(ninjaidxt1mu0)=acd261(77)
      brack(ninjaidxt0mu0)=acd261(78)
      brack(ninjaidxt0mu2)=0.0_ki
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d261h8_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd261h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k4
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d261h8l131
