module     p0_ubaru_httbar_d77h14l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity14d77h14l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd77h14_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(19) :: acd77
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd77(1)=dotproduct(ninjaE3,spvak2k1)
      acd77(2)=dotproduct(ninjaE3,spvak2l3)
      acd77(3)=abb77(10)
      acd77(4)=dotproduct(ninjaE3,spval3l5)
      acd77(5)=abb77(15)
      acd77(6)=dotproduct(ninjaE3,spvak2l4)
      acd77(7)=abb77(21)
      acd77(8)=dotproduct(ninjaE3,spval3l4)
      acd77(9)=abb77(18)
      acd77(10)=dotproduct(ninjaE3,spvak2l5)
      acd77(11)=abb77(20)
      acd77(12)=dotproduct(ninjaE3,spval3k1)
      acd77(13)=abb77(16)
      acd77(14)=abb77(26)
      acd77(15)=acd77(3)*acd77(2)
      acd77(16)=acd77(5)*acd77(4)
      acd77(17)=acd77(7)*acd77(6)
      acd77(18)=acd77(9)*acd77(8)
      acd77(19)=acd77(11)*acd77(10)
      acd77(15)=acd77(19)+acd77(18)+acd77(17)+acd77(15)+acd77(16)
      acd77(15)=acd77(1)*acd77(15)
      acd77(16)=acd77(13)*acd77(6)
      acd77(17)=acd77(14)*acd77(10)
      acd77(16)=acd77(17)+acd77(16)
      acd77(16)=acd77(12)*acd77(16)
      acd77(15)=acd77(15)+acd77(16)
      brack(ninjaidxt2mu0)=acd77(15)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd77h14_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(47) :: acd77
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd77(1)=dotproduct(ninjaE3,spvak2k1)
      acd77(2)=dotproduct(ninjaE4,spvak2l3)
      acd77(3)=abb77(10)
      acd77(4)=dotproduct(ninjaE4,spvak2l4)
      acd77(5)=abb77(21)
      acd77(6)=dotproduct(ninjaE4,spval3l5)
      acd77(7)=abb77(15)
      acd77(8)=dotproduct(ninjaE4,spval3l4)
      acd77(9)=abb77(18)
      acd77(10)=dotproduct(ninjaE4,spvak2l5)
      acd77(11)=abb77(20)
      acd77(12)=dotproduct(ninjaE3,spvak2l3)
      acd77(13)=dotproduct(ninjaE4,spvak2k1)
      acd77(14)=dotproduct(ninjaE3,spvak2l4)
      acd77(15)=dotproduct(ninjaE4,spval3k1)
      acd77(16)=abb77(16)
      acd77(17)=dotproduct(ninjaE3,spval3l5)
      acd77(18)=dotproduct(ninjaE3,spval3k1)
      acd77(19)=abb77(26)
      acd77(20)=dotproduct(ninjaE3,spval3l4)
      acd77(21)=dotproduct(ninjaE3,spvak2l5)
      acd77(22)=abb77(13)
      acd77(23)=dotproduct(ninjaA,ninjaE3)
      acd77(24)=dotproduct(ninjaA,spvak2k1)
      acd77(25)=dotproduct(ninjaA,spvak2l3)
      acd77(26)=dotproduct(ninjaA,spvak2l4)
      acd77(27)=dotproduct(ninjaA,spval3l5)
      acd77(28)=dotproduct(ninjaA,spval3k1)
      acd77(29)=dotproduct(ninjaA,spval3l4)
      acd77(30)=dotproduct(ninjaA,spvak2l5)
      acd77(31)=abb77(12)
      acd77(32)=abb77(11)
      acd77(33)=abb77(22)
      acd77(34)=abb77(25)
      acd77(35)=dotproduct(ninjaA,ninjaA)
      acd77(36)=abb77(19)
      acd77(37)=acd77(11)*acd77(10)
      acd77(38)=acd77(9)*acd77(8)
      acd77(39)=acd77(7)*acd77(6)
      acd77(40)=acd77(5)*acd77(4)
      acd77(41)=acd77(3)*acd77(2)
      acd77(37)=acd77(41)+acd77(37)+acd77(38)+acd77(39)+acd77(40)
      acd77(37)=acd77(37)*acd77(1)
      acd77(38)=acd77(11)*acd77(21)
      acd77(39)=acd77(9)*acd77(20)
      acd77(40)=acd77(7)*acd77(17)
      acd77(41)=acd77(5)*acd77(14)
      acd77(42)=acd77(3)*acd77(12)
      acd77(38)=acd77(42)+acd77(38)+acd77(39)+acd77(40)+acd77(41)
      acd77(39)=acd77(38)*acd77(13)
      acd77(40)=acd77(14)*acd77(16)
      acd77(41)=acd77(19)*acd77(21)
      acd77(40)=acd77(40)+acd77(41)
      acd77(40)=acd77(40)*acd77(15)
      acd77(42)=acd77(10)*acd77(18)*acd77(19)
      acd77(43)=acd77(16)*acd77(18)
      acd77(44)=acd77(43)*acd77(4)
      acd77(37)=acd77(37)+acd77(42)+acd77(40)+acd77(39)+acd77(44)+acd77(22)
      acd77(38)=acd77(24)*acd77(38)
      acd77(39)=acd77(11)*acd77(30)
      acd77(40)=acd77(9)*acd77(29)
      acd77(42)=acd77(7)*acd77(27)
      acd77(44)=acd77(5)*acd77(26)
      acd77(45)=acd77(3)*acd77(25)
      acd77(39)=acd77(31)+acd77(42)+acd77(39)+acd77(40)+acd77(44)+acd77(45)
      acd77(40)=acd77(1)*acd77(39)
      acd77(42)=acd77(22)*acd77(23)
      acd77(44)=acd77(21)*acd77(34)
      acd77(41)=acd77(28)*acd77(41)
      acd77(45)=acd77(19)*acd77(30)
      acd77(45)=acd77(45)+acd77(33)
      acd77(46)=acd77(18)*acd77(45)
      acd77(43)=acd77(26)*acd77(43)
      acd77(47)=acd77(16)*acd77(28)
      acd77(47)=acd77(32)+acd77(47)
      acd77(47)=acd77(14)*acd77(47)
      acd77(38)=acd77(40)+acd77(38)+acd77(47)+acd77(43)+acd77(46)+acd77(41)+2.0&
      &_ki*acd77(42)+acd77(44)
      acd77(40)=ninjaP*acd77(37)
      acd77(39)=acd77(24)*acd77(39)
      acd77(41)=acd77(16)*acd77(26)
      acd77(41)=acd77(41)+acd77(45)
      acd77(41)=acd77(28)*acd77(41)
      acd77(42)=acd77(22)*acd77(35)
      acd77(43)=acd77(30)*acd77(34)
      acd77(44)=acd77(26)*acd77(32)
      acd77(39)=acd77(39)+acd77(44)+acd77(43)+acd77(36)+acd77(42)+acd77(41)+acd&
      &77(40)
      brack(ninjaidxt1mu0)=acd77(38)
      brack(ninjaidxt0mu0)=acd77(39)
      brack(ninjaidxt0mu2)=acd77(37)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d77h14_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd77h14_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
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
end module     p0_ubaru_httbar_d77h14l131_qp
