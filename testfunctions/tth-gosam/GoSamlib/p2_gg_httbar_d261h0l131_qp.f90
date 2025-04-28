module     p2_gg_httbar_d261h0l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d261h0l131_qp.f90
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
      use p2_gg_httbar_abbrevd261h0_qp
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd261h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(71) :: acd261
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd261(1)=dotproduct(ninjaE3,spvae1k2)
      acd261(2)=dotproduct(ninjaE3,spval5e2)
      acd261(3)=dotproduct(ninjaE3,spvae2e1)
      acd261(4)=abb261(47)
      acd261(5)=dotproduct(ninjaE3,spvae2k2)
      acd261(6)=dotproduct(ninjaE3,spvae1e2)
      acd261(7)=dotproduct(ninjaE3,spval4e1)
      acd261(8)=abb261(65)
      acd261(9)=dotproduct(ninjaA,ninjaE3)
      acd261(10)=abb261(33)
      acd261(11)=dotproduct(ninjaE3,spval4l3)
      acd261(12)=abb261(56)
      acd261(13)=dotproduct(ninjaE3,spvak2e1)
      acd261(14)=abb261(11)
      acd261(15)=abb261(24)
      acd261(16)=abb261(59)
      acd261(17)=abb261(15)
      acd261(18)=dotproduct(ninjaE3,spval3e1)
      acd261(19)=abb261(50)
      acd261(20)=dotproduct(ninjaE3,spval4k2)
      acd261(21)=abb261(19)
      acd261(22)=dotproduct(ninjaE3,spval5k2)
      acd261(23)=abb261(21)
      acd261(24)=abb261(34)
      acd261(25)=dotproduct(ninjaE3,spval5l3)
      acd261(26)=abb261(38)
      acd261(27)=dotproduct(ninjaE3,spvae2k1)
      acd261(28)=abb261(48)
      acd261(29)=dotproduct(ninjaA,spvae1k2)
      acd261(30)=dotproduct(ninjaA,spvae2k2)
      acd261(31)=dotproduct(ninjaA,spval5e2)
      acd261(32)=dotproduct(ninjaA,spvae2e1)
      acd261(33)=dotproduct(ninjaA,spvae1e2)
      acd261(34)=dotproduct(ninjaA,spval4e1)
      acd261(35)=abb261(7)
      acd261(36)=abb261(17)
      acd261(37)=abb261(10)
      acd261(38)=abb261(14)
      acd261(39)=abb261(43)
      acd261(40)=abb261(49)
      acd261(41)=abb261(41)
      acd261(42)=abb261(16)
      acd261(43)=abb261(22)
      acd261(44)=abb261(60)
      acd261(45)=abb261(31)
      acd261(46)=abb261(62)
      acd261(47)=abb261(26)
      acd261(48)=dotproduct(ninjaE3,spvae1l3)
      acd261(49)=abb261(45)
      acd261(50)=abb261(32)
      acd261(51)=abb261(54)
      acd261(52)=abb261(53)
      acd261(53)=abb261(52)
      acd261(54)=dotproduct(ninjaE3,spval4e2)
      acd261(55)=abb261(23)
      acd261(56)=abb261(37)
      acd261(57)=abb261(46)
      acd261(58)=acd261(5)*acd261(7)*acd261(6)*acd261(8)
      acd261(59)=acd261(1)*acd261(2)*acd261(3)*acd261(4)
      acd261(58)=acd261(58)+acd261(59)
      acd261(59)=acd261(22)*acd261(23)
      acd261(60)=acd261(20)*acd261(21)
      acd261(61)=acd261(27)*acd261(28)
      acd261(62)=acd261(25)*acd261(26)
      acd261(63)=acd261(11)*acd261(12)
      acd261(64)=acd261(13)*acd261(14)
      acd261(65)=acd261(7)*acd261(24)
      acd261(66)=acd261(2)*acd261(17)
      acd261(67)=acd261(18)*acd261(19)
      acd261(68)=acd261(5)*acd261(16)
      acd261(69)=acd261(1)*acd261(15)
      acd261(70)=2.0_ki*acd261(9)
      acd261(71)=acd261(10)*acd261(70)
      acd261(59)=acd261(71)+acd261(69)+acd261(68)+acd261(67)+acd261(66)+acd261(&
      &65)+acd261(64)+acd261(63)+acd261(62)+acd261(61)+acd261(59)+acd261(60)
      acd261(59)=acd261(59)*acd261(70)
      acd261(60)=acd261(22)*acd261(51)
      acd261(61)=acd261(20)*acd261(50)
      acd261(62)=acd261(27)*acd261(53)
      acd261(63)=acd261(25)*acd261(52)
      acd261(64)=acd261(11)*acd261(36)
      acd261(60)=acd261(64)+acd261(63)+acd261(62)+acd261(60)+acd261(61)
      acd261(60)=acd261(18)*acd261(60)
      acd261(61)=acd261(8)*acd261(34)
      acd261(61)=acd261(45)+acd261(61)
      acd261(61)=acd261(6)*acd261(61)
      acd261(62)=acd261(13)*acd261(38)
      acd261(63)=acd261(8)*acd261(33)
      acd261(63)=acd261(46)+acd261(63)
      acd261(63)=acd261(7)*acd261(63)
      acd261(64)=acd261(18)*acd261(44)
      acd261(61)=acd261(64)+acd261(63)+acd261(61)+acd261(62)
      acd261(61)=acd261(5)*acd261(61)
      acd261(62)=acd261(13)*acd261(37)
      acd261(63)=acd261(4)*acd261(31)
      acd261(63)=acd261(43)+acd261(63)
      acd261(63)=acd261(3)*acd261(63)
      acd261(64)=acd261(4)*acd261(32)
      acd261(64)=acd261(41)+acd261(64)
      acd261(64)=acd261(2)*acd261(64)
      acd261(65)=acd261(18)*acd261(42)
      acd261(62)=acd261(65)+acd261(64)+acd261(62)+acd261(63)
      acd261(62)=acd261(1)*acd261(62)
      acd261(63)=acd261(27)*acd261(40)
      acd261(64)=acd261(25)*acd261(39)
      acd261(65)=acd261(11)*acd261(35)
      acd261(63)=acd261(65)+acd261(63)+acd261(64)
      acd261(63)=acd261(13)*acd261(63)
      acd261(64)=acd261(48)*acd261(57)
      acd261(65)=acd261(8)*acd261(30)
      acd261(65)=acd261(56)+acd261(65)
      acd261(65)=acd261(6)*acd261(65)
      acd261(64)=acd261(64)+acd261(65)
      acd261(64)=acd261(7)*acd261(64)
      acd261(65)=acd261(48)*acd261(49)
      acd261(66)=acd261(4)*acd261(29)
      acd261(66)=acd261(47)+acd261(66)
      acd261(66)=acd261(3)*acd261(66)
      acd261(65)=acd261(65)+acd261(66)
      acd261(65)=acd261(2)*acd261(65)
      acd261(66)=acd261(3)*acd261(54)*acd261(55)
      acd261(59)=acd261(59)+acd261(62)+acd261(61)+acd261(60)+acd261(65)+acd261(&
      &64)+acd261(63)+acd261(66)
      brack(ninjaidxt1mu0)=acd261(58)
      brack(ninjaidxt0mu0)=acd261(59)
      brack(ninjaidxt0mu2)=0.0_ki
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d261h0_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd261h0_qp
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
end module     p2_gg_httbar_d261h0l131_qp
