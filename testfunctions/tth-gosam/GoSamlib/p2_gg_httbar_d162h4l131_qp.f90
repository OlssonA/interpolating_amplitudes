module     p2_gg_httbar_d162h4l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d162h4l131_qp.f90
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
      use p2_gg_httbar_abbrevd162h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(24) :: acd162
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd162(1)=dotproduct(ninjaE3,spvak1e1)
      acd162(2)=dotproduct(ninjaE3,spvae1k2)
      acd162(3)=abb162(13)
      acd162(4)=dotproduct(ninjaE3,spval5e1)
      acd162(5)=abb162(37)
      acd162(6)=dotproduct(ninjaE3,spvae2e1)
      acd162(7)=abb162(23)
      acd162(8)=dotproduct(ninjaE3,spvak2e1)
      acd162(9)=abb162(26)
      acd162(10)=dotproduct(ninjaE3,spval4e1)
      acd162(11)=abb162(43)
      acd162(12)=dotproduct(ninjaE3,spvae1e2)
      acd162(13)=abb162(17)
      acd162(14)=dotproduct(ninjaE3,spvae1l5)
      acd162(15)=abb162(27)
      acd162(16)=dotproduct(ninjaE3,spvae1k1)
      acd162(17)=abb162(30)
      acd162(18)=dotproduct(ninjaE3,spvae1l4)
      acd162(19)=abb162(39)
      acd162(20)=acd162(5)*acd162(2)
      acd162(21)=acd162(13)*acd162(12)
      acd162(22)=acd162(15)*acd162(14)
      acd162(23)=acd162(17)*acd162(16)
      acd162(24)=acd162(19)*acd162(18)
      acd162(20)=acd162(24)+acd162(23)+acd162(22)+acd162(21)+acd162(20)
      acd162(20)=acd162(4)*acd162(20)
      acd162(21)=acd162(3)*acd162(1)
      acd162(22)=-acd162(7)*acd162(6)
      acd162(23)=acd162(9)*acd162(8)
      acd162(24)=-acd162(11)*acd162(10)
      acd162(21)=acd162(24)+acd162(23)+acd162(22)+acd162(21)
      acd162(21)=acd162(2)*acd162(21)
      acd162(20)=acd162(20)+acd162(21)
      brack(ninjaidxt2mu0)=acd162(20)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd162h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(65) :: acd162
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd162(1)=dotproduct(ninjaE3,spvak1e1)
      acd162(2)=dotproduct(ninjaE4,spvae1k2)
      acd162(3)=abb162(13)
      acd162(4)=dotproduct(ninjaE3,spvae1k2)
      acd162(5)=dotproduct(ninjaE4,spvak1e1)
      acd162(6)=dotproduct(ninjaE4,spval5e1)
      acd162(7)=abb162(37)
      acd162(8)=dotproduct(ninjaE4,spvae2e1)
      acd162(9)=abb162(23)
      acd162(10)=dotproduct(ninjaE4,spvak2e1)
      acd162(11)=abb162(26)
      acd162(12)=dotproduct(ninjaE4,spval4e1)
      acd162(13)=abb162(43)
      acd162(14)=dotproduct(ninjaE3,spval5e1)
      acd162(15)=dotproduct(ninjaE4,spvae1e2)
      acd162(16)=abb162(17)
      acd162(17)=dotproduct(ninjaE4,spvae1k1)
      acd162(18)=abb162(30)
      acd162(19)=dotproduct(ninjaE4,spvae1l5)
      acd162(20)=abb162(27)
      acd162(21)=dotproduct(ninjaE4,spvae1l4)
      acd162(22)=abb162(39)
      acd162(23)=dotproduct(ninjaE3,spvae1e2)
      acd162(24)=dotproduct(ninjaE3,spvae2e1)
      acd162(25)=dotproduct(ninjaE3,spvae1k1)
      acd162(26)=dotproduct(ninjaE3,spvak2e1)
      acd162(27)=dotproduct(ninjaE3,spvae1l5)
      acd162(28)=dotproduct(ninjaE3,spvae1l4)
      acd162(29)=dotproduct(ninjaE3,spval4e1)
      acd162(30)=dotproduct(ninjaA,spvak1e1)
      acd162(31)=dotproduct(ninjaA,spvae1k2)
      acd162(32)=dotproduct(ninjaA,spval5e1)
      acd162(33)=dotproduct(ninjaA,spvae1e2)
      acd162(34)=dotproduct(ninjaA,spvae2e1)
      acd162(35)=dotproduct(ninjaA,spvae1k1)
      acd162(36)=dotproduct(ninjaA,spvak2e1)
      acd162(37)=dotproduct(ninjaA,spvae1l5)
      acd162(38)=dotproduct(ninjaA,spvae1l4)
      acd162(39)=dotproduct(ninjaA,spval4e1)
      acd162(40)=abb162(16)
      acd162(41)=abb162(15)
      acd162(42)=abb162(14)
      acd162(43)=abb162(21)
      acd162(44)=abb162(19)
      acd162(45)=abb162(20)
      acd162(46)=abb162(25)
      acd162(47)=abb162(33)
      acd162(48)=abb162(29)
      acd162(49)=abb162(35)
      acd162(50)=abb162(12)
      acd162(51)=acd162(22)*acd162(21)
      acd162(52)=acd162(20)*acd162(19)
      acd162(53)=acd162(18)*acd162(17)
      acd162(54)=acd162(16)*acd162(15)
      acd162(55)=acd162(2)*acd162(7)
      acd162(51)=acd162(51)+acd162(53)+acd162(54)+acd162(52)+acd162(55)
      acd162(51)=acd162(51)*acd162(14)
      acd162(52)=acd162(13)*acd162(12)
      acd162(53)=acd162(11)*acd162(10)
      acd162(54)=acd162(9)*acd162(8)
      acd162(55)=acd162(3)*acd162(5)
      acd162(56)=acd162(6)*acd162(7)
      acd162(52)=-acd162(56)+acd162(52)-acd162(53)+acd162(54)-acd162(55)
      acd162(52)=acd162(52)*acd162(4)
      acd162(53)=acd162(13)*acd162(29)
      acd162(54)=acd162(11)*acd162(26)
      acd162(55)=acd162(9)*acd162(24)
      acd162(56)=acd162(3)*acd162(1)
      acd162(53)=-acd162(56)+acd162(55)+acd162(53)-acd162(54)
      acd162(54)=acd162(53)*acd162(2)
      acd162(55)=acd162(22)*acd162(28)
      acd162(56)=acd162(20)*acd162(27)
      acd162(57)=acd162(18)*acd162(25)
      acd162(58)=acd162(16)*acd162(23)
      acd162(55)=acd162(55)+acd162(56)+acd162(57)+acd162(58)
      acd162(56)=acd162(55)*acd162(6)
      acd162(51)=-acd162(54)+acd162(56)+acd162(51)-acd162(52)
      acd162(52)=acd162(13)*acd162(39)
      acd162(54)=acd162(11)*acd162(36)
      acd162(56)=acd162(9)*acd162(34)
      acd162(57)=acd162(3)*acd162(30)
      acd162(58)=acd162(32)*acd162(7)
      acd162(52)=-acd162(52)+acd162(54)-acd162(56)+acd162(57)+acd162(58)+acd162&
      &(41)
      acd162(54)=acd162(4)*acd162(52)
      acd162(55)=acd162(32)*acd162(55)
      acd162(53)=-acd162(31)*acd162(53)
      acd162(56)=acd162(22)*acd162(38)
      acd162(57)=acd162(20)*acd162(37)
      acd162(58)=acd162(18)*acd162(35)
      acd162(59)=acd162(16)*acd162(33)
      acd162(56)=acd162(56)+acd162(57)+acd162(58)+acd162(59)+acd162(42)
      acd162(57)=acd162(31)*acd162(7)
      acd162(57)=acd162(57)+acd162(56)
      acd162(57)=acd162(14)*acd162(57)
      acd162(58)=acd162(29)*acd162(49)
      acd162(59)=acd162(28)*acd162(48)
      acd162(60)=acd162(27)*acd162(47)
      acd162(61)=acd162(26)*acd162(46)
      acd162(62)=acd162(25)*acd162(45)
      acd162(63)=acd162(24)*acd162(44)
      acd162(64)=acd162(23)*acd162(43)
      acd162(65)=acd162(1)*acd162(40)
      acd162(53)=acd162(54)+acd162(57)+acd162(53)+acd162(55)+acd162(65)+acd162(&
      &64)+acd162(63)+acd162(62)+acd162(61)+acd162(60)+acd162(58)+acd162(59)
      acd162(54)=ninjaP*acd162(51)
      acd162(52)=acd162(31)*acd162(52)
      acd162(55)=acd162(32)*acd162(56)
      acd162(56)=acd162(39)*acd162(49)
      acd162(57)=acd162(38)*acd162(48)
      acd162(58)=acd162(37)*acd162(47)
      acd162(59)=acd162(36)*acd162(46)
      acd162(60)=acd162(35)*acd162(45)
      acd162(61)=acd162(34)*acd162(44)
      acd162(62)=acd162(33)*acd162(43)
      acd162(63)=acd162(30)*acd162(40)
      acd162(52)=acd162(54)+acd162(52)+acd162(55)+acd162(63)+acd162(62)+acd162(&
      &61)+acd162(60)+acd162(59)+acd162(58)+acd162(57)+acd162(50)+acd162(56)
      brack(ninjaidxt1mu0)=acd162(53)
      brack(ninjaidxt0mu0)=acd162(52)
      brack(ninjaidxt0mu2)=acd162(51)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d162h4_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd162h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
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
end module     p2_gg_httbar_d162h4l131_qp
