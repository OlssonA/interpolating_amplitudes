module     p2_gg_httbar_d142h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d142h8l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd142h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc142(43)
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: QspQ
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      QspQ = dotproduct(Q,Q)
      acc142(1)=abb142(12)
      acc142(2)=abb142(13)
      acc142(3)=abb142(14)
      acc142(4)=abb142(15)
      acc142(5)=abb142(16)
      acc142(6)=abb142(17)
      acc142(7)=abb142(18)
      acc142(8)=abb142(19)
      acc142(9)=abb142(20)
      acc142(10)=abb142(21)
      acc142(11)=abb142(22)
      acc142(12)=abb142(23)
      acc142(13)=abb142(24)
      acc142(14)=abb142(25)
      acc142(15)=abb142(26)
      acc142(16)=abb142(27)
      acc142(17)=abb142(31)
      acc142(18)=abb142(34)
      acc142(19)=abb142(37)
      acc142(20)=abb142(53)
      acc142(21)=abb142(59)
      acc142(22)=abb142(70)
      acc142(23)=abb142(71)
      acc142(24)=abb142(72)
      acc142(25)=abb142(78)
      acc142(26)=abb142(85)
      acc142(27)=abb142(87)
      acc142(28)=abb142(95)
      acc142(29)=abb142(103)
      acc142(30)=abb142(108)
      acc142(31)=-Qspvae2l4*acc142(30)
      acc142(32)=acc142(14)*Qspvae2k1
      acc142(33)=acc142(19)*Qspvae2k2
      acc142(34)=acc142(27)*Qspvae2l5
      acc142(35)=acc142(28)*Qspvae2e1
      acc142(31)=acc142(35)+acc142(34)+acc142(24)+acc142(33)+acc142(32)+acc142(&
      &31)
      acc142(31)=Qspval4e2*acc142(31)
      acc142(32)=acc142(1)*Qspvae2k2
      acc142(33)=acc142(2)*Qspvae2k1
      acc142(34)=acc142(9)*Qspvae2e1
      acc142(35)=acc142(10)*Qspvae2l5
      acc142(36)=acc142(11)*Qspvae2l4
      acc142(32)=acc142(36)+acc142(35)+acc142(34)+acc142(6)+acc142(33)+acc142(3&
      &2)
      acc142(32)=Qspvak2e2*acc142(32)
      acc142(33)=acc142(5)*Qspvae2e1
      acc142(34)=acc142(17)*Qspvae2k2
      acc142(35)=acc142(26)*Qspvae2l5
      acc142(33)=acc142(35)+acc142(22)+acc142(34)+acc142(33)
      acc142(33)=Qspvae1e2*acc142(33)
      acc142(34)=-acc142(30)*Qspvae2k1
      acc142(35)=acc142(15)*Qspvae2l5
      acc142(36)=acc142(16)*Qspvae2k2
      acc142(34)=acc142(36)+acc142(35)+acc142(3)+acc142(34)
      acc142(34)=Qspvak1e2*acc142(34)
      acc142(35)=-acc142(30)*Qspvae2l5
      acc142(36)=acc142(18)*Qspvae2k2
      acc142(35)=acc142(29)+acc142(36)+acc142(35)
      acc142(35)=Qspval5e2*acc142(35)
      acc142(36)=acc142(8)*Qspvae2k1
      acc142(37)=acc142(13)*Qspvae2k2
      acc142(38)=acc142(20)*Qspvae2e1
      acc142(39)=acc142(23)*Qspvae2l5
      acc142(40)=acc142(25)*Qspvae2l4
      acc142(41)=Qspvae2l3*acc142(12)
      acc142(42)=Qspval3e2*acc142(4)
      acc142(43)=QspQ*acc142(7)
      brack=acc142(21)+acc142(31)+acc142(32)+acc142(33)+acc142(34)+acc142(35)+a&
      &cc142(36)+acc142(37)+acc142(38)+acc142(39)+acc142(40)+acc142(41)+acc142(&
      &42)+acc142(43)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d142h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd142h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d142
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d142 = 0.0_ki
      d142 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d142, ki), aimag(d142), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d142h8l1_qp
