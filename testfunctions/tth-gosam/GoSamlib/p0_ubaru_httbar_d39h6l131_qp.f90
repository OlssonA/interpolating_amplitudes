module     p0_ubaru_httbar_d39h6l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d39h6l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd39h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(43) :: acd39
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd39(1)=dotproduct(k1,ninjaE3)
      acd39(2)=abb39(26)
      acd39(3)=dotproduct(k2,ninjaA)
      acd39(4)=dotproduct(ninjaE3,spvak2k1)
      acd39(5)=abb39(12)
      acd39(6)=dotproduct(k2,ninjaE3)
      acd39(7)=dotproduct(ninjaA,spvak2k1)
      acd39(8)=abb39(59)
      acd39(9)=dotproduct(ninjaA,ninjaE3)
      acd39(10)=abb39(25)
      acd39(11)=dotproduct(ninjaE3,spval5l3)
      acd39(12)=abb39(11)
      acd39(13)=dotproduct(ninjaE3,spval5l4)
      acd39(14)=abb39(13)
      acd39(15)=dotproduct(ninjaE3,spval3l4)
      acd39(16)=abb39(14)
      acd39(17)=dotproduct(ninjaE3,spval3k2)
      acd39(18)=abb39(15)
      acd39(19)=dotproduct(ninjaE3,spvak2l3)
      acd39(20)=abb39(16)
      acd39(21)=dotproduct(ninjaA,spval5l3)
      acd39(22)=dotproduct(ninjaA,spval5l4)
      acd39(23)=dotproduct(ninjaA,spval3l4)
      acd39(24)=dotproduct(ninjaA,spval3k2)
      acd39(25)=dotproduct(ninjaA,spvak2l3)
      acd39(26)=abb39(10)
      acd39(27)=abb39(35)
      acd39(28)=dotproduct(ninjaE3,spval3k1)
      acd39(29)=abb39(17)
      acd39(30)=dotproduct(ninjaE3,spvak2l4)
      acd39(31)=abb39(29)
      acd39(32)=dotproduct(ninjaE3,spval5k1)
      acd39(33)=abb39(40)
      acd39(34)=acd39(3)*acd39(5)
      acd39(35)=acd39(21)*acd39(12)
      acd39(36)=acd39(22)*acd39(14)
      acd39(37)=acd39(23)*acd39(16)
      acd39(38)=acd39(24)*acd39(18)
      acd39(39)=acd39(25)*acd39(20)
      acd39(34)=acd39(26)+acd39(39)+acd39(38)+acd39(37)+acd39(36)+acd39(35)+acd&
      &39(34)
      acd39(34)=acd39(4)*acd39(34)
      acd39(35)=acd39(6)*acd39(5)
      acd39(36)=acd39(20)*acd39(19)
      acd39(37)=acd39(11)*acd39(12)
      acd39(38)=acd39(13)*acd39(14)
      acd39(39)=acd39(15)*acd39(16)
      acd39(40)=acd39(17)*acd39(18)
      acd39(35)=acd39(35)+acd39(36)+acd39(37)+acd39(38)+acd39(39)+acd39(40)
      acd39(36)=acd39(7)*acd39(35)
      acd39(37)=acd39(2)*acd39(1)
      acd39(38)=acd39(8)*acd39(6)
      acd39(39)=acd39(10)*acd39(9)
      acd39(40)=acd39(27)*acd39(19)
      acd39(41)=acd39(29)*acd39(28)
      acd39(42)=acd39(31)*acd39(30)
      acd39(43)=acd39(33)*acd39(32)
      acd39(34)=acd39(43)+acd39(42)+acd39(41)+acd39(40)+2.0_ki*acd39(39)+acd39(&
      &38)+acd39(37)+acd39(36)+acd39(34)
      acd39(35)=acd39(4)*acd39(35)
      brack(ninjaidxt3mu0)=acd39(35)
      brack(ninjaidxt2mu0)=acd39(34)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d39h6_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd39h6_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k4-k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d39h6l131_qp
