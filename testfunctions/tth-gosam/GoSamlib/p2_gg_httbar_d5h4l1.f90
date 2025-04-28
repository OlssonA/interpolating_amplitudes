module     p2_gg_httbar_d5h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d5h4l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd5h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc5(31)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc5(1)=abb5(6)
      acc5(2)=abb5(7)
      acc5(3)=abb5(8)
      acc5(4)=abb5(9)
      acc5(5)=abb5(10)
      acc5(6)=abb5(11)
      acc5(7)=abb5(12)
      acc5(8)=abb5(13)
      acc5(9)=abb5(14)
      acc5(10)=abb5(15)
      acc5(11)=abb5(16)
      acc5(12)=abb5(17)
      acc5(13)=abb5(18)
      acc5(14)=abb5(19)
      acc5(15)=abb5(20)
      acc5(16)=abb5(21)
      acc5(17)=Qspvae2e1*acc5(10)
      acc5(18)=Qspvae1e2*acc5(1)
      acc5(19)=Qspvae2l3*acc5(11)
      acc5(20)=Qspval3e2*acc5(14)
      acc5(21)=Qspvae1l3*acc5(4)
      acc5(22)=Qspval3e1*acc5(7)
      acc5(23)=Qspvak2e2*acc5(8)
      acc5(24)=Qspvak2e1*acc5(3)
      acc5(25)=Qspval5l4*acc5(2)
      acc5(26)=Qspval5l3*acc5(13)
      acc5(27)=Qspval3l4*acc5(16)
      acc5(28)=Qspval3k2*acc5(15)
      acc5(29)=Qspvak2l3*acc5(9)
      acc5(30)=Qspk2*acc5(5)
      acc5(31)=QspQ*acc5(6)
      brack=acc5(12)+acc5(17)+acc5(18)+acc5(19)+acc5(20)+acc5(21)+acc5(22)+acc5&
      &(23)+acc5(24)+acc5(25)+acc5(26)+acc5(27)+acc5(28)+acc5(29)+acc5(30)+acc5&
      &(31)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d5h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd5h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d5
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d5 = 0.0_ki
      d5 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d5, ki), aimag(d5), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d5h4l1
