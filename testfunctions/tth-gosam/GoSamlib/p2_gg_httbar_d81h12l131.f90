module     p2_gg_httbar_d81h12l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d81h12l131.f90
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
      use p2_gg_httbar_abbrevd81h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(15) :: acd81
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd81(1)=dotproduct(ninjaE3,spvak2e2)
      acd81(2)=dotproduct(ninjaE3,spvae1l4)
      acd81(3)=dotproduct(ninjaE3,spvae2e1)
      acd81(4)=abb81(9)
      acd81(5)=dotproduct(ninjaE3,spvae1l3)
      acd81(6)=abb81(33)
      acd81(7)=dotproduct(ninjaE3,spvak2e1)
      acd81(8)=dotproduct(ninjaE3,spvae2l5)
      acd81(9)=dotproduct(ninjaE3,spvae1e2)
      acd81(10)=abb81(10)
      acd81(11)=dotproduct(ninjaE3,spval3e1)
      acd81(12)=abb81(38)
      acd81(13)=acd81(10)*acd81(7)
      acd81(14)=-acd81(12)*acd81(11)
      acd81(13)=acd81(14)+acd81(13)
      acd81(13)=acd81(13)*acd81(9)*acd81(8)
      acd81(14)=acd81(4)*acd81(2)
      acd81(15)=acd81(6)*acd81(5)
      acd81(14)=acd81(14)+acd81(15)
      acd81(14)=acd81(14)*acd81(3)*acd81(1)
      acd81(13)=acd81(14)+acd81(13)
      brack(ninjaidxt2mu0)=acd81(13)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd81h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(89) :: acd81
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd81(1)=dotproduct(ninjaE3,spvak2e2)
      acd81(2)=dotproduct(ninjaE3,spvae1l4)
      acd81(3)=dotproduct(ninjaE4,spvae2e1)
      acd81(4)=abb81(9)
      acd81(5)=dotproduct(ninjaE3,spvae2e1)
      acd81(6)=dotproduct(ninjaE4,spvae1l4)
      acd81(7)=dotproduct(ninjaE4,spvae1l3)
      acd81(8)=abb81(33)
      acd81(9)=dotproduct(ninjaE3,spvae1l3)
      acd81(10)=dotproduct(ninjaE4,spvak2e2)
      acd81(11)=dotproduct(ninjaE3,spvak2e1)
      acd81(12)=dotproduct(ninjaE3,spvae2l5)
      acd81(13)=dotproduct(ninjaE4,spvae1e2)
      acd81(14)=abb81(10)
      acd81(15)=dotproduct(ninjaE3,spvae1e2)
      acd81(16)=dotproduct(ninjaE4,spvae2l5)
      acd81(17)=dotproduct(ninjaE4,spvak2e1)
      acd81(18)=dotproduct(ninjaE4,spval3e1)
      acd81(19)=abb81(38)
      acd81(20)=dotproduct(ninjaE3,spval3e1)
      acd81(21)=dotproduct(ninjaA,spvak2e2)
      acd81(22)=dotproduct(ninjaA,spvae1l4)
      acd81(23)=dotproduct(ninjaA,spvae2e1)
      acd81(24)=dotproduct(ninjaA,spvak2e1)
      acd81(25)=dotproduct(ninjaA,spvae2l5)
      acd81(26)=dotproduct(ninjaA,spvae1e2)
      acd81(27)=dotproduct(ninjaA,spval3e1)
      acd81(28)=dotproduct(ninjaA,spvae1l3)
      acd81(29)=abb81(23)
      acd81(30)=dotproduct(ninjaE3,spval3e2)
      acd81(31)=abb81(43)
      acd81(32)=abb81(16)
      acd81(33)=abb81(18)
      acd81(34)=dotproduct(ninjaE3,spvae2l3)
      acd81(35)=abb81(39)
      acd81(36)=abb81(45)
      acd81(37)=dotproduct(ninjaE3,spvae2l4)
      acd81(38)=abb81(25)
      acd81(39)=dotproduct(k2,ninjaE3)
      acd81(40)=abb81(11)
      acd81(41)=dotproduct(ninjaA,ninjaE3)
      acd81(42)=abb81(13)
      acd81(43)=dotproduct(ninjaA,spvae2l3)
      acd81(44)=dotproduct(ninjaA,spvae2l4)
      acd81(45)=dotproduct(ninjaA,spval3e2)
      acd81(46)=abb81(14)
      acd81(47)=abb81(31)
      acd81(48)=abb81(32)
      acd81(49)=abb81(17)
      acd81(50)=abb81(28)
      acd81(51)=abb81(44)
      acd81(52)=abb81(12)
      acd81(53)=dotproduct(ninjaE3,spvak1e2)
      acd81(54)=abb81(15)
      acd81(55)=dotproduct(ninjaE3,spvae2k1)
      acd81(56)=abb81(19)
      acd81(57)=dotproduct(ninjaE3,spval3k2)
      acd81(58)=abb81(20)
      acd81(59)=abb81(21)
      acd81(60)=abb81(22)
      acd81(61)=dotproduct(ninjaE3,spvak2l4)
      acd81(62)=abb81(27)
      acd81(63)=abb81(30)
      acd81(64)=dotproduct(ninjaE3,spvak2l5)
      acd81(65)=abb81(42)
      acd81(66)=dotproduct(ninjaE3,spvak2l3)
      acd81(67)=abb81(46)
      acd81(68)=acd81(8)*acd81(7)
      acd81(69)=acd81(4)*acd81(6)
      acd81(68)=acd81(68)+acd81(69)
      acd81(68)=acd81(68)*acd81(1)
      acd81(69)=acd81(8)*acd81(9)
      acd81(70)=acd81(4)*acd81(2)
      acd81(69)=acd81(69)+acd81(70)
      acd81(70)=acd81(10)*acd81(69)
      acd81(70)=acd81(68)+acd81(70)
      acd81(70)=acd81(5)*acd81(70)
      acd81(71)=acd81(19)*acd81(18)
      acd81(72)=acd81(14)*acd81(17)
      acd81(71)=acd81(71)-acd81(72)
      acd81(71)=acd81(71)*acd81(12)
      acd81(72)=acd81(19)*acd81(20)
      acd81(73)=acd81(14)*acd81(11)
      acd81(72)=acd81(72)-acd81(73)
      acd81(73)=-acd81(16)*acd81(72)
      acd81(73)=-acd81(71)+acd81(73)
      acd81(73)=acd81(15)*acd81(73)
      acd81(74)=acd81(72)*acd81(12)
      acd81(75)=-acd81(13)*acd81(74)
      acd81(76)=acd81(69)*acd81(1)
      acd81(77)=acd81(3)*acd81(76)
      acd81(70)=acd81(73)+acd81(70)+acd81(75)+acd81(77)
      acd81(72)=acd81(72)*acd81(25)
      acd81(73)=acd81(37)*acd81(38)
      acd81(75)=acd81(34)*acd81(35)
      acd81(77)=acd81(20)*acd81(36)
      acd81(78)=acd81(11)*acd81(32)
      acd81(72)=-acd81(72)+acd81(73)+acd81(75)+acd81(77)+acd81(78)
      acd81(73)=-acd81(19)*acd81(27)
      acd81(75)=acd81(14)*acd81(24)
      acd81(73)=acd81(75)+acd81(33)+acd81(73)
      acd81(73)=acd81(12)*acd81(73)
      acd81(73)=acd81(73)+acd81(72)
      acd81(73)=acd81(15)*acd81(73)
      acd81(69)=acd81(69)*acd81(21)
      acd81(75)=acd81(31)*acd81(30)
      acd81(69)=acd81(75)+acd81(69)
      acd81(75)=acd81(8)*acd81(28)
      acd81(77)=acd81(4)*acd81(22)
      acd81(75)=acd81(77)+acd81(29)+acd81(75)
      acd81(75)=acd81(1)*acd81(75)
      acd81(75)=acd81(75)+acd81(69)
      acd81(75)=acd81(5)*acd81(75)
      acd81(74)=-acd81(26)*acd81(74)
      acd81(76)=acd81(23)*acd81(76)
      acd81(73)=acd81(73)+acd81(75)+acd81(74)+acd81(76)
      acd81(74)=-acd81(25)*acd81(27)
      acd81(75)=ninjaP*acd81(20)
      acd81(76)=-acd81(16)*acd81(75)
      acd81(74)=acd81(74)+acd81(76)
      acd81(74)=acd81(19)*acd81(74)
      acd81(76)=acd81(25)*acd81(24)
      acd81(77)=ninjaP*acd81(11)
      acd81(78)=acd81(16)*acd81(77)
      acd81(76)=acd81(76)+acd81(78)
      acd81(76)=acd81(14)*acd81(76)
      acd81(71)=-ninjaP*acd81(71)
      acd81(78)=acd81(38)*acd81(44)
      acd81(79)=acd81(35)*acd81(43)
      acd81(80)=acd81(27)*acd81(36)
      acd81(81)=acd81(24)*acd81(32)
      acd81(82)=acd81(25)*acd81(33)
      acd81(71)=acd81(71)+acd81(76)+acd81(74)+acd81(82)+acd81(81)+acd81(80)+acd&
      &81(79)+acd81(51)+acd81(78)
      acd81(71)=acd81(15)*acd81(71)
      acd81(72)=acd81(26)*acd81(72)
      acd81(74)=acd81(21)*acd81(28)
      acd81(76)=ninjaP*acd81(9)
      acd81(78)=acd81(10)*acd81(76)
      acd81(74)=acd81(74)+acd81(78)
      acd81(74)=acd81(8)*acd81(74)
      acd81(78)=acd81(21)*acd81(22)
      acd81(79)=ninjaP*acd81(2)
      acd81(80)=acd81(10)*acd81(79)
      acd81(78)=acd81(78)+acd81(80)
      acd81(78)=acd81(4)*acd81(78)
      acd81(68)=ninjaP*acd81(68)
      acd81(80)=acd81(31)*acd81(45)
      acd81(81)=acd81(21)*acd81(29)
      acd81(68)=acd81(68)+acd81(78)+acd81(74)+acd81(81)+acd81(48)+acd81(80)
      acd81(68)=acd81(5)*acd81(68)
      acd81(74)=-acd81(26)*acd81(27)
      acd81(75)=-acd81(13)*acd81(75)
      acd81(74)=acd81(74)+acd81(75)
      acd81(74)=acd81(19)*acd81(74)
      acd81(75)=acd81(26)*acd81(24)
      acd81(77)=acd81(13)*acd81(77)
      acd81(75)=acd81(75)+acd81(77)
      acd81(75)=acd81(14)*acd81(75)
      acd81(77)=acd81(26)*acd81(33)
      acd81(74)=acd81(75)+acd81(74)+acd81(50)+acd81(77)
      acd81(74)=acd81(12)*acd81(74)
      acd81(75)=acd81(23)*acd81(28)
      acd81(76)=acd81(3)*acd81(76)
      acd81(75)=acd81(75)+acd81(76)
      acd81(75)=acd81(8)*acd81(75)
      acd81(76)=acd81(23)*acd81(22)
      acd81(77)=acd81(3)*acd81(79)
      acd81(76)=acd81(76)+acd81(77)
      acd81(76)=acd81(4)*acd81(76)
      acd81(77)=acd81(23)*acd81(29)
      acd81(75)=acd81(76)+acd81(75)+acd81(46)+acd81(77)
      acd81(75)=acd81(1)*acd81(75)
      acd81(69)=acd81(23)*acd81(69)
      acd81(76)=acd81(66)*acd81(67)
      acd81(77)=-acd81(64)*acd81(65)
      acd81(78)=acd81(61)*acd81(62)
      acd81(79)=acd81(57)*acd81(58)
      acd81(80)=acd81(55)*acd81(56)
      acd81(81)=acd81(53)*acd81(54)
      acd81(82)=acd81(41)*acd81(42)
      acd81(83)=acd81(39)*acd81(40)
      acd81(84)=acd81(37)*acd81(60)
      acd81(85)=acd81(34)*acd81(52)
      acd81(86)=-acd81(9)*acd81(63)
      acd81(87)=-acd81(2)*acd81(47)
      acd81(88)=acd81(20)*acd81(59)
      acd81(89)=acd81(11)*acd81(49)
      acd81(68)=acd81(71)+acd81(68)+acd81(75)+acd81(74)+acd81(89)+acd81(88)+acd&
      &81(87)+acd81(86)+acd81(85)+acd81(84)+acd81(83)+2.0_ki*acd81(82)+acd81(81&
      &)+acd81(80)+acd81(79)+acd81(78)+acd81(76)+acd81(77)+acd81(72)+acd81(69)
      brack(ninjaidxt1mu0)=acd81(73)
      brack(ninjaidxt0mu0)=acd81(68)
      brack(ninjaidxt0mu2)=acd81(70)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d81h12_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd81h12
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
end module     p2_gg_httbar_d81h12l131
