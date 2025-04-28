module     p2_gg_httbar_d79h4l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d79h4l131.f90
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
      use p2_gg_httbar_abbrevd79h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd79
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd79h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(94) :: acd79
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd79(1)=dotproduct(ninjaE3,spvae1l4)
      acd79(2)=abb79(10)
      acd79(3)=dotproduct(ninjaE3,spvae1l5)
      acd79(4)=abb79(11)
      acd79(5)=dotproduct(ninjaE3,spvae1e2)
      acd79(6)=abb79(12)
      acd79(7)=dotproduct(ninjaE3,spvak2e1)
      acd79(8)=abb79(26)
      acd79(9)=dotproduct(ninjaE3,spvae2e1)
      acd79(10)=abb79(27)
      acd79(11)=dotproduct(ninjaE3,spval5e1)
      acd79(12)=abb79(39)
      acd79(13)=dotproduct(k2,ninjaE3)
      acd79(14)=abb79(32)
      acd79(15)=dotproduct(ninjaA,ninjaE3)
      acd79(16)=abb79(25)
      acd79(17)=abb79(30)
      acd79(18)=abb79(31)
      acd79(19)=dotproduct(ninjaE3,spval3l4)
      acd79(20)=dotproduct(ninjaE3,spvae1l3)
      acd79(21)=dotproduct(ninjaE3,spval3l5)
      acd79(22)=dotproduct(ninjaE3,spval3e2)
      acd79(23)=abb79(28)
      acd79(24)=abb79(15)
      acd79(25)=dotproduct(ninjaE3,spvae1k2)
      acd79(26)=dotproduct(ninjaE3,spvak2l4)
      acd79(27)=abb79(18)
      acd79(28)=dotproduct(ninjaE3,spvak2e2)
      acd79(29)=abb79(23)
      acd79(30)=dotproduct(ninjaE3,spvak2l5)
      acd79(31)=abb79(35)
      acd79(32)=dotproduct(ninjaE3,spvae2k2)
      acd79(33)=abb79(33)
      acd79(34)=dotproduct(ninjaE3,spval5k2)
      acd79(35)=abb79(34)
      acd79(36)=dotproduct(ninjaE3,spval3e1)
      acd79(37)=dotproduct(ninjaE3,spvak2l3)
      acd79(38)=dotproduct(ninjaE3,spvae2l3)
      acd79(39)=dotproduct(ninjaE3,spval5l3)
      acd79(40)=dotproduct(k2,ninjaA)
      acd79(41)=dotproduct(ninjaA,spvak2e1)
      acd79(42)=dotproduct(ninjaA,ninjaA)
      acd79(43)=dotproduct(ninjaA,spvae1l4)
      acd79(44)=dotproduct(ninjaA,spvae1l5)
      acd79(45)=dotproduct(ninjaA,spvae1e2)
      acd79(46)=dotproduct(ninjaA,spvae2e1)
      acd79(47)=dotproduct(ninjaA,spval5e1)
      acd79(48)=abb79(14)
      acd79(49)=dotproduct(ninjaA,spval3l4)
      acd79(50)=dotproduct(ninjaA,spvae1l3)
      acd79(51)=dotproduct(ninjaA,spval3l5)
      acd79(52)=dotproduct(ninjaA,spval3e2)
      acd79(53)=dotproduct(ninjaA,spvae1k2)
      acd79(54)=dotproduct(ninjaA,spvak2l4)
      acd79(55)=dotproduct(ninjaA,spval3e1)
      acd79(56)=dotproduct(ninjaA,spvak2e2)
      acd79(57)=dotproduct(ninjaA,spvak2l3)
      acd79(58)=dotproduct(ninjaA,spvae2l3)
      acd79(59)=dotproduct(ninjaA,spvae2k2)
      acd79(60)=dotproduct(ninjaA,spval5k2)
      acd79(61)=dotproduct(ninjaA,spvak2l5)
      acd79(62)=dotproduct(ninjaA,spval5l3)
      acd79(63)=abb79(29)
      acd79(64)=abb79(24)
      acd79(65)=abb79(21)
      acd79(66)=abb79(17)
      acd79(67)=abb79(13)
      acd79(68)=abb79(16)
      acd79(69)=abb79(19)
      acd79(70)=abb79(20)
      acd79(71)=abb79(22)
      acd79(72)=acd79(2)*acd79(1)
      acd79(73)=acd79(8)*acd79(7)
      acd79(74)=acd79(3)*acd79(4)
      acd79(75)=acd79(5)*acd79(6)
      acd79(76)=acd79(9)*acd79(10)
      acd79(77)=acd79(11)*acd79(12)
      acd79(72)=acd79(72)-acd79(73)-acd79(74)-acd79(75)+acd79(76)-acd79(77)
      acd79(73)=-acd79(15)*acd79(72)
      acd79(74)=acd79(23)*acd79(3)
      acd79(75)=acd79(24)*acd79(5)
      acd79(76)=acd79(13)*acd79(14)
      acd79(77)=acd79(32)*acd79(33)
      acd79(78)=acd79(34)*acd79(35)
      acd79(74)=acd79(78)+acd79(74)+acd79(75)+acd79(76)+acd79(77)
      acd79(75)=acd79(16)*acd79(1)
      acd79(75)=acd79(75)+acd79(74)
      acd79(75)=acd79(7)*acd79(75)
      acd79(76)=acd79(19)*acd79(2)
      acd79(77)=acd79(21)*acd79(4)
      acd79(78)=acd79(22)*acd79(6)
      acd79(76)=-acd79(78)+acd79(76)-acd79(77)
      acd79(77)=-acd79(20)*acd79(76)
      acd79(78)=acd79(26)*acd79(27)
      acd79(79)=acd79(28)*acd79(29)
      acd79(80)=acd79(30)*acd79(31)
      acd79(78)=acd79(80)+acd79(78)+acd79(79)
      acd79(79)=acd79(25)*acd79(78)
      acd79(80)=acd79(37)*acd79(8)
      acd79(81)=acd79(38)*acd79(10)
      acd79(82)=acd79(39)*acd79(12)
      acd79(80)=acd79(82)+acd79(80)-acd79(81)
      acd79(81)=acd79(36)*acd79(80)
      acd79(82)=acd79(17)*acd79(9)
      acd79(83)=acd79(18)*acd79(11)
      acd79(82)=acd79(82)+acd79(83)
      acd79(83)=acd79(1)*acd79(82)
      acd79(73)=2.0_ki*acd79(73)+acd79(75)+acd79(81)+acd79(79)+acd79(77)+acd79(&
      &83)
      acd79(75)=-ninjaP-acd79(42)
      acd79(75)=acd79(72)*acd79(75)
      acd79(77)=2.0_ki*acd79(15)
      acd79(79)=acd79(8)*acd79(77)
      acd79(74)=acd79(79)+acd79(74)
      acd79(74)=acd79(41)*acd79(74)
      acd79(76)=-acd79(50)*acd79(76)
      acd79(78)=acd79(53)*acd79(78)
      acd79(79)=acd79(55)*acd79(80)
      acd79(80)=acd79(40)*acd79(14)
      acd79(81)=acd79(59)*acd79(33)
      acd79(83)=acd79(60)*acd79(35)
      acd79(80)=acd79(68)+acd79(83)+acd79(81)+acd79(80)
      acd79(80)=acd79(7)*acd79(80)
      acd79(81)=-acd79(49)*acd79(2)
      acd79(83)=acd79(51)*acd79(4)
      acd79(84)=acd79(52)*acd79(6)
      acd79(81)=acd79(64)+acd79(84)+acd79(83)+acd79(81)
      acd79(81)=acd79(20)*acd79(81)
      acd79(83)=acd79(54)*acd79(27)
      acd79(84)=acd79(56)*acd79(29)
      acd79(85)=acd79(61)*acd79(31)
      acd79(83)=acd79(67)+acd79(85)+acd79(84)+acd79(83)
      acd79(83)=acd79(25)*acd79(83)
      acd79(84)=acd79(57)*acd79(8)
      acd79(85)=-acd79(58)*acd79(10)
      acd79(86)=acd79(62)*acd79(12)
      acd79(84)=acd79(69)+acd79(86)+acd79(85)+acd79(84)
      acd79(84)=acd79(36)*acd79(84)
      acd79(85)=-acd79(2)*acd79(77)
      acd79(82)=acd79(85)+acd79(82)
      acd79(82)=acd79(43)*acd79(82)
      acd79(85)=acd79(41)*acd79(1)
      acd79(86)=acd79(43)*acd79(7)
      acd79(85)=acd79(85)+acd79(86)
      acd79(85)=acd79(16)*acd79(85)
      acd79(86)=acd79(4)*acd79(77)
      acd79(87)=acd79(23)*acd79(7)
      acd79(86)=acd79(86)+acd79(87)
      acd79(86)=acd79(44)*acd79(86)
      acd79(87)=acd79(6)*acd79(77)
      acd79(88)=acd79(24)*acd79(7)
      acd79(87)=acd79(87)+acd79(88)
      acd79(87)=acd79(45)*acd79(87)
      acd79(88)=-acd79(10)*acd79(77)
      acd79(89)=acd79(17)*acd79(1)
      acd79(88)=acd79(88)+acd79(89)
      acd79(88)=acd79(46)*acd79(88)
      acd79(89)=acd79(12)*acd79(77)
      acd79(90)=acd79(18)*acd79(1)
      acd79(89)=acd79(89)+acd79(90)
      acd79(89)=acd79(47)*acd79(89)
      acd79(77)=acd79(48)*acd79(77)
      acd79(90)=acd79(63)*acd79(1)
      acd79(91)=acd79(65)*acd79(3)
      acd79(92)=acd79(66)*acd79(5)
      acd79(93)=acd79(70)*acd79(9)
      acd79(94)=acd79(71)*acd79(11)
      acd79(74)=acd79(94)+acd79(93)+acd79(92)+acd79(91)+acd79(90)+acd79(77)+acd&
      &79(89)+acd79(88)+acd79(87)+acd79(86)+acd79(85)+acd79(74)+acd79(84)+acd79&
      &(83)+acd79(81)+acd79(80)+acd79(79)+acd79(78)+acd79(76)+acd79(82)+acd79(7&
      &5)
      brack(ninjaidxt1mu0)=acd79(73)
      brack(ninjaidxt0mu0)=acd79(74)
      brack(ninjaidxt0mu2)=-acd79(72)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d79h4_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd79h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4
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
end module     p2_gg_httbar_d79h4l131
